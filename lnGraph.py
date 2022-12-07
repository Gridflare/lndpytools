import json
import os
import time

import networkx as nx
import igraph

from nodeinterface import NodeInterface


class lnGraphBase:
    """Holds common methods for the below classes"""

    @classmethod
    def autoload(cls, expirehours=8):
        """Intelligently load from a json file or node"""

        # Check for json, check age
        graphfilename = 'describegraph.json'
        if os.path.isfile(graphfilename):
            mtime = os.path.getmtime(graphfilename)

            # if expired, warn and exit
            if expirehours:
                if time.time() - mtime > expirehours * 60 * 60:
                    print(graphfilename, 'was found but is more than 8 hours old')
                    print('Please update it or delete to attempt fetching from lnd')
                    exit()

            return cls.fromjson(graphfile=graphfilename)

        else:
            # fromconfig will create and exit if the config is missing
            ni = NodeInterface.fromconfig()

            # else load from lnd
            print('Fetching graph data from lnd')
            return cls.fromlnd(lndnode=ni)

class gRPCadapters:
    """Collection of useful gRPC to JSON conversions"""

    @staticmethod
    def addrs2dict(addrs):
        # explicit convert to Python types
        addrlist = []
        for addr in addrs:
            addrdict = {'network': addr.network, 'addr': addr.addr}
            addrlist.append(addrdict)
        return addrlist

    @staticmethod
    def nodepolicy2dict(np):
        return dict(
            time_lock_delta=np.time_lock_delta,
            min_htlc=np.min_htlc,
            fee_base_msat=np.fee_base_msat,
            fee_rate_milli_msat=np.fee_rate_milli_msat,
            disabled=np.disabled,
            max_htlc_msat=np.max_htlc_msat,
            last_update=np.last_update,
        )


class lnGraph(lnGraphBase, nx.Graph):
    """Methods for loading network data into networkx"""

    @classmethod
    def fromjson(cls, graphfile='describegraph.json'):
        with open(graphfile, encoding='utf8') as f:
            graphdata = json.load(f)

        g = cls()
        # Leave weight undefined for now, could be capacity or fee depending on algorithm
        for node in graphdata['nodes']:
            g.add_node(node['pub_key'],
                       pub_key=node['pub_key'],
                       last_update=node['last_update'],
                       alias=node['alias'],
                       color=node['color'],
                       addresses=node['addresses'],
                       capacity=0,  # Increment while iterating channels
                       num_channels=0,  # Increment while iterating channels
                       capacities=[],  # For creating histograms
                       )

        for edge in graphdata['edges']:
            n1 = edge['node1_pub']
            n2 = edge['node2_pub']
            cap = int(edge['capacity'])

            # track node capacity
            # Note that this will not account for channels filtered later
            for n in [n1, n2]:
                g.nodes[n]['capacity'] += cap
                g.nodes[n]['capacities'].append(cap)
                g.nodes[n]['num_channels'] += 1

            edgeparams = dict(
                channel_id=edge['channel_id'],
                chan_point=edge['chan_point'],
                last_update=edge['last_update'],
                capacity=cap,
                node1_pub=n1,
                node2_pub=n2,
                node1_policy=edge['node1_policy'],
                node2_policy=edge['node2_policy'],
            )

            redundant_edges = []
            if g.has_edge(n1, n2):

                # Track the total value between nodes
                link_capacity = cap + g.edges[n1, n2]['link_capacity']

                # Need to decide to overwrite or hide
                if g.edges[n1, n2]['capacity'] < cap:
                    # Old edge is smaller, replace
                    lesser_edge = g.edges[n1, n2].copy()
                    redundant_edges = lesser_edge['redundant_edges']
                    del lesser_edge['redundant_edges']
                    del lesser_edge['link_capacity']
                    redundant_edges.append(lesser_edge)
                else:
                    # Old edge is bigger, keep it
                    g.edges[n1, n2]['redundant_edges'].append(edgeparams)
                    continue
            else:
                link_capacity = cap


            edgeparams['redundant_edges'] = redundant_edges
            edgeparams['link_capacity'] = link_capacity

            g.add_edge(n1, n2, **edgeparams)

            if redundant_edges:
                assert g.edges[n1, n2]['channel_id'] != redundant_edges[0]['channel_id']

        return g

    @classmethod
    def fromlnd(cls, lndnode: NodeInterface = None, include_unannounced=False):
        if lndnode is None:
            lndnode = NodeInterface()  # use defaults

        graphdata = lndnode.DescribeGraph(include_unannounced=include_unannounced)

        g = cls()

        # Leave weight undefined for now, could be capacity or fee depending on algorithm
        for node in graphdata.nodes:
            g.add_node(node.pub_key,
                       pub_key=node.pub_key,
                       last_update=node.last_update,
                       alias=node.alias,
                       color=node.color,
                       addresses=gRPCadapters.addrs2dict(node.addresses),
                       capacity=0,  # Increment while iterating channels
                       num_channels=0,  # Increment while iterating channels
                       capacities=[],  # For easily creating histograms of channels
                       )

        for edge in graphdata.edges:
            n1 = edge.node1_pub
            n2 = edge.node2_pub
            cap = edge.capacity

            # track node capacity
            # Note that this will not account for channels filtered later
            for n in [n1, n2]:
                g.nodes[n]['capacity'] += cap
                g.nodes[n]['capacities'].append(cap)
                g.nodes[n]['num_channels'] += 1


            edgeparams = dict(
                channel_id=edge.channel_id,
                chan_point=edge.chan_point,
                last_update=edge.last_update,
                capacity=cap,
                node1_pub=n1,
                node2_pub=n2,
                node1_policy=gRPCadapters.nodepolicy2dict(edge.node1_policy),
                node2_policy=gRPCadapters.nodepolicy2dict(edge.node2_policy),
            )

            redundant_edges = []
            if g.has_edge(n1, n2):

                # Track the total value between nodes
                link_capacity = cap + g.edges[n1, n2]['link_capacity']

                # Need to decide to overwrite or hide
                if g.edges[n1, n2]['capacity'] < cap:
                    # Old edge is smaller, replace
                    lesser_edge = g.edges[n1, n2].copy()
                    redundant_edges = lesser_edge['redundant_edges']
                    del lesser_edge['redundant_edges']
                    del lesser_edge['link_capacity']
                    redundant_edges.append(lesser_edge)
                else:
                    # Old edge is bigger, keep it
                    g.edges[n1, n2]['redundant_edges'].append(edgeparams)
                    continue
            else:
                link_capacity = cap

            edgeparams['redundant_edges'] = redundant_edges
            edgeparams['link_capacity'] = cap

            g.add_edge(n1, n2, **edgeparams)

            if redundant_edges:  # sanity check for reference leaks
                assert g.edges[n1, n2]['channel_id'] != redundant_edges[0]['channel_id']

        return g

    def channels(self, nodeid):
        """Return channels for a node, including redundants"""

        channels = []
        for peerkey in self.adj[mynodekey]:
            mainchan = self.edges[peerkey, mynodekey].copy()
            redundants = mainchan['redundant_edges']
            del mainchan['redundant_edges']
            channels.append(mainchan)
            channels.extend(redundants)

        return channels


class lnGraphV2(lnGraphBase, igraph.Graph):

    @classmethod
    def fromjson(cls, graphfile='describegraph.json'):
        with open(graphfile, encoding='utf8') as f:
            graphdata = json.load(f)

        # Directed graph will be superior when fee metrics are computed
        g = cls(directed=True)

        intattrs = ['capacity', 'last_update',
                    'fee_base_msat', 'fee_rate_milli_msat',
                    'min_htlc', 'max_htlc_msat']

        def fixstr2int(data: dict):
            if data is None: return None
            for k, v in data.items():
                if k in intattrs:
                    data[k] = int(v)
            return data

        nodeattrs = {a:[] for a in ['pub_key', 'last_update', 'alias', 'color', 'addresses']}
        for node in graphdata['nodes']:
            node = fixstr2int(node)
            for k in nodeattrs.keys():
                nodeattrs[k].append(node[k])



        n = len(nodeattrs['pub_key'])
        nodeattrs['capacity'] = [0]*n  # Increment while iterating channels
        nodeattrs['num_channels'] = [0]*n  # Increment while iterating channels

        edgeattrs = {a:[] for a in ['channel_id', 'chan_point', 'capacity', 'last_update']}
        policies1 = []
        policies2 = []
        edgeids1 = []
        edgeids2 = []
        nodepubs1 = []
        nodepubs2 = []
        nodeindexmap = {k:i for i, k in enumerate(nodeattrs['pub_key'])}

        for edge in graphdata['edges']:
            edge = fixstr2int(edge)
            n1pub = edge['node1_pub']
            n2pub = edge['node2_pub']
            cap = edge['capacity']

            for k in edgeattrs.keys():
                edgeattrs[k].append(edge[k])

            edgeids1.append((n1pub,n2pub))
            edgeids2.append((n2pub,n1pub))
            nodepubs1.append(n1pub)
            nodepubs2.append(n2pub)

            n1i = nodeindexmap[n1pub]
            n2i = nodeindexmap[n2pub]

            nodeattrs['capacity'][n1i] += cap
            nodeattrs['capacity'][n2i] += cap
            nodeattrs['num_channels'][n1i] += 1
            nodeattrs['num_channels'][n2i] += 1

            policies1.append(fixstr2int(edge['node1_policy']))
            policies2.append(fixstr2int(edge['node2_policy']))

        g.add_vertices(nodeattrs['pub_key'], nodeattrs)

        # Using a directed graph, have to add edges twice because channels are bidirectionsl
        edgeattrs['policy_out'] = policies1
        edgeattrs['policy_in'] = policies2
        edgeattrs['local_pubkey'] = nodepubs1
        edgeattrs['remote_pubkey'] = nodepubs2

        g.add_edges(edgeids1, edgeattrs)

        edgeattrs['policy_out'] = policies2
        edgeattrs['policy_in'] = policies1
        edgeattrs['local_pubkey'] = nodepubs2
        edgeattrs['remote_pubkey'] = nodepubs1

        g.add_edges(edgeids2, edgeattrs)
        return g

    @classmethod
    def fromlnd(cls, lndnode: NodeInterface = None, include_unannounced=False):
        if lndnode is None:
            lndnode = NodeInterface()  # use defaults

        graphdata = lndnode.DescribeGraph(include_unannounced=include_unannounced)

        g = cls(directed=True)

        nodeattrs = {a:[] for a in ['pub_key', 'last_update', 'alias', 'color', 'addresses']}
        for node in graphdata.nodes:
            for k in nodeattrs.keys():
                a = getattr(node, k)
                if k == 'addresses':
                    a = gRPCadapters.addrs2dict(a)
                nodeattrs[k].append(a)

        n = len(nodeattrs['pub_key']) # Number of nodes in graph
        nodeattrs['capacity'] = [0]*n  # Increment while iterating channels
        nodeattrs['num_channels'] = [0]*n  # Increment while iterating channels

        edgeattrs = {a:[] for a in ['channel_id', 'chan_point', 'capacity', 'last_update']}
        policies1 = []
        policies2 = []
        edgeids1 = []
        edgeids2 = []
        nodepubs1 = []
        nodepubs2 = []
        nodeindexmap = {k:i for i, k in enumerate(nodeattrs['pub_key'])}

        for edge in graphdata.edges:
            n1pub = edge.node1_pub
            n2pub = edge.node2_pub
            cap = int(edge.capacity)

            for k in edgeattrs.keys():
                edgeattrs[k].append(getattr(edge, k))

            edgeids1.append((n1pub,n2pub))
            edgeids2.append((n2pub,n1pub))
            nodepubs1.append(n1pub)
            nodepubs2.append(n2pub)

            n1i = nodeindexmap[n1pub]
            n2i = nodeindexmap[n2pub]

            nodeattrs['capacity'][n1i] += cap
            nodeattrs['capacity'][n2i] += cap
            nodeattrs['num_channels'][n1i] += 1
            nodeattrs['num_channels'][n2i] += 1

            policies1.append(gRPCadapters.nodepolicy2dict(edge.node1_policy))
            policies2.append(gRPCadapters.nodepolicy2dict(edge.node2_policy))

        g.add_vertices(nodeattrs['pub_key'], nodeattrs)

        # Using a directed graph, have to add edges twice to keep track of policies
        edgeattrs['policy_out'] = policies1
        edgeattrs['policy_in'] = policies2
        edgeattrs['local_pubkey'] = n1pub
        edgeattrs['remote_pubkey'] = n2pub

        g.add_edges(edgeids1, edgeattrs)

        edgeattrs['policy_out'] = policies2
        edgeattrs['policy_in'] = policies1
        edgeattrs['local_pubkey'] = n2pub
        edgeattrs['remote_pubkey'] = n1pub

        g.add_edges(edgeids2, edgeattrs)
        return g

    @property
    def nodes(self):
        return self.vs

    @property
    def channels(self):
        return self.es

