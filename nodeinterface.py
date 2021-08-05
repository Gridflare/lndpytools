#!/usr/bin/env python3
"""
This is a wrapper around LND's gRPC interface

It is incomplete and largely untested, read-only usage is highly recommended

"""

import os
import configparser
from functools import lru_cache
import codecs

import grpc

from lnrpc_generated import rpc_pb2 as ln, rpc_pb2_grpc as lnrpc
from lnrpc_generated.walletrpc import walletkit_pb2 as walletrpc, walletkit_pb2_grpc as walletkitstub


MSGMAXMB = 50 * 1024 * 1024
LNDDIR = os.path.expanduser('~/.lnd')

class BaseInterface:
    """A class that tries to intelligently call functions from an LND service
    This introspection does not work in many cases.
    """

    def __getattr__(self, cmd):
        """
        Some magic for undefined functions, QOL hack
        """

        if hasattr(self._rpc, cmd+'Request'):
            lnfunc = getattr(self._rpc, cmd+'Request')
        elif  hasattr(self._rpc, f'Get{cmd}Request'):
            lnfunc = getattr(self._rpc, f'Get{cmd}Request')
        else:
            raise NotImplementedError('Unhandled method self._rpc.(Get)' + cmd + 'Request')

        if hasattr(self._stub, cmd):
            stubfunc = getattr(self._stub, cmd)

            def rpcCommand(*args,**kwargs):
                return stubfunc(lnfunc(*args, **kwargs))
            return rpcCommand

        elif hasattr(self._stub, 'Get'+cmd):
            stubfunc = getattr(self._stub, 'Get'+cmd)
            def rpcCommand(*args,**kwargs):
                if args:
                    raise TypeError('Cannot use positional arguments with this command')
                return stubfunc(lnfunc(**kwargs))
            return rpcCommand

        else:
            raise NotImplementedError('Unhandled method stub.(Get)' + cmd)

class SubserverRPC(BaseInterface):
    """
    Generic class for subservers, may need to be extended in the future
    """
    def __init__(self, subRPC, substub):
        self._rpc = subRPC
        self._stub = substub

class MinimalNodeInterface(BaseInterface):
    """A class implementing the bare minimum to communicate with LND over RPC"""

    def __init__(self, server=None, tlspath=None, macpath=None, cachedir='_cache'):

        if server is None: server = 'localhost:10009'
        if tlspath is None: tlspath = LNDDIR + '/tls.cert'
        if macpath is None: macpath = LNDDIR + '/data/chain/bitcoin/mainnet/admin.macaroon'

        assert os.path.isfile(tlspath), tlspath + ' does not exist!'
        assert os.path.isfile(macpath), macpath + ' does not exist!'
        assert tlspath.endswith(('.cert','.crt'))
        assert macpath.endswith('.macaroon')

        tlsCert = open(tlspath, 'rb').read()
        sslCred = grpc.ssl_channel_credentials(tlsCert)
        macaroon = codecs.encode(open(macpath, 'rb').read(), 'hex')
        authCred = grpc.metadata_call_credentials(
                        lambda _, callback: callback(
                            [('macaroon', macaroon)], None))
        masterCred = grpc.composite_channel_credentials(sslCred, authCred)

        os.environ['GRPC_SSL_CIPHER_SUITES'] = 'HIGH+ECDSA'

        options = [
                ('grpc.max_message_length', MSGMAXMB),
                ('grpc.max_receive_message_length', MSGMAXMB)
            ]

        grpc_channel = grpc.secure_channel(server, masterCred, options)

        self._stub = lnrpc.LightningStub(grpc_channel)
        self._rpc = ln

        self.wallet = SubserverRPC(walletrpc, walletkitstub.WalletKitStub(grpc_channel))

        if not os.path.exists(cachedir):
            os.mkdir(cachedir)

        self.cachedir = cachedir

    @classmethod
    def fromconfig(cls, conffile='node.conf', nodename='Node1'):
        if not os.path.isfile(conffile):
            print('Config for lnd not found, will create', conffile)
            config = configparser.ConfigParser()
            config[nodename] = {'server': 'localhost:10009',
                'macpath': LNDDIR + '/data/chain/bitcoin/mainnet/readonly.macaroon',
                'tlspath': LNDDIR + '/tls.cert',
                }

            with open(conffile, 'w') as cf:
                config.write(cf)

            print('Please check/complete the config and rerun')
            print('If running remotely you will need a local copy of your macaroon and tls.cert')
            print('Using readonly.macaroon is recommended unless you know what you are doing.')
            exit()

        else:
            config = configparser.ConfigParser()
            config.read(conffile)
            return cls(**config[nodename])


class BasicNodeInterface(MinimalNodeInterface):
    """A subclass of MinimalNodeInterface implementing missing methods"""

    def UpdateChannelPolicy(self, **kwargs):
        if 'chan_point' in kwargs and isinstance(kwargs['chan_point'], str):
            cp = kwargs['chan_point']
            kwargs['chan_point'] = ln.ChannelPoint(
                funding_txid_str=cp.split(':')[0],
                output_index=int(cp.split(':')[1])
            )

        return self._stub.UpdateChannelPolicy(ln.PolicyUpdateRequest(**kwargs))

    def DescribeGraph(self, include_unannounced=True):
        return self._stub.DescribeGraph(
            ln.ChannelGraphRequest(include_unannounced=include_unannounced))

    def getForwardingHistory(self, starttime):
        """Same as the bare metal method, but this one pages automatically"""
        fwdhist = []
        offset = 0

        def getfwdbatch(starttime, pageOffset):
            fwdbatch = self.ForwardingHistory(
                        start_time=starttime,
                        index_offset=pageOffset,
                        ).forwarding_events
            return fwdbatch

        while (fwdbatch := getfwdbatch(starttime, offset)):
            offset += len(fwdbatch)
            fwdhist.extend(fwdbatch)

        return fwdhist

    def ListInvoices(self, **kwargs):
        """Wrapper required due to inconsistent naming"""
        return self._stub.ListInvoices(ln.ListInvoiceRequest(**kwargs))


class AdvancedNodeInterface(BasicNodeInterface):
    """Class implementing recombinant methods not directly available from LND"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

    @lru_cache(maxsize=256)
    def getAlias(self, pubkey=None):
        if pubkey is None:
            return self.GetInfo().alias
        else:
            return self._stub.GetNodeInfo(ln.NodeInfoRequest(pub_key=pubkey)).node.alias

    def getNeighboursInfo(self,pubkey=None):
        """Return more useful info about our, or another node's, neighbours"""
        return list(map(self.GetNodeInfo, self.getNeighboursPubkeys(pubkey)))


class NodeInterface(AdvancedNodeInterface):
    """
    A class streamlining the LND RPC interface
    Alias for AdvancedNodeInterface
    Methods that are forwarded directly to LND are capitalised e.g. GetInfo()
    Methods that process the data in any way use lowerCamelCase e.g. getAlias()
    """

if __name__ == '__main__':
    # Testing connectivity
    ni = NodeInterface.fromconfig()
    print('Connected to node', ni.getAlias())




