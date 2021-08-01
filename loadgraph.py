
import json
import os
import time

import networkx as nx

from nodeinterface import NodeInterface

# Multigraph is the most honest representation,
# but in practice we only care about the largest channels
# Some algorithms also prefer the simpler graph

class lnGraph(nx.Graph):
    """Extension of networkx Graph to handle parallel channels"""

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
                       capacity=0, # Increment while iterating channels
                       num_channels=0, # Increment while iterating channels
                       capacities=[], # For creating histograms
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

                # Need to decide to overwrite or hide
                if g.edges[n1, n2]['capacity'] < cap:
                    # Old edge is smaller, replace
                    lesser_edge = g.edges[n1, n2].copy()
                    redundant_edges = lesser_edge['redundant_edges']
                    del lesser_edge['redundant_edges']
                    redundant_edges.append(lesser_edge)
                else:
                    #Old edge is bigger, keep it
                    g.edges[n1, n2]['redundant_edges'].append(edgeparams)
                    continue

            edgeparams['redundant_edges']=redundant_edges

            g.add_edge(n1, n2, **edgeparams)

            if redundant_edges:
                assert g.edges[n1, n2]['channel_id'] != redundant_edges[0]['channel_id']

        return g

    @classmethod
    def fromlnd(cls, lndnode:NodeInterface=None, include_unannounced=False):
        if lndnode is None:
            lndnode = NodeInterface() # use defaults

        graphdata = lndnode.DescribeGraph(include_unannounced=include_unannounced)

        g = cls()

        def convertAddrs2py(addrs):
            # explicit convert to Python types
            addrlist = []
            for addr in addrs:
                addrdict = {'network':addr.network, 'addr':addr.addr}
                addrlist.append(addrdict)
            return addrlist

        # Leave weight undefined for now, could be capacity or fee depending on algorithm
        for node in graphdata.nodes:
            g.add_node(node.pub_key,
                       pub_key=node.pub_key,
                       last_update=node.last_update,
                       alias=node.alias,
                       color=node.color,
                       addresses=convertAddrs2py(node.addresses),
                       capacity=0, # Increment while iterating channels
                       num_channels=0, # Increment while iterating channels
                       capacities=[], # For easily creating histograms of channels
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

            edgeparams = dict(
                           channel_id=edge.channel_id,
                           chan_point=edge.chan_point,
                           last_update=edge.last_update,
                           capacity=cap,
                           node1_pub=n1,
                           node2_pub=n2,
                           node1_policy=nodepolicy2dict(edge.node1_policy),
                           node2_policy=nodepolicy2dict(edge.node2_policy),
                           )

            redundant_edges = []
            if g.has_edge(n1, n2):

                # Need to decide to overwrite or hide
                if g.edges[n1, n2]['capacity'] < cap:
                    # Old edge is smaller, replace
                    lesser_edge = g.edges[n1, n2].copy()
                    redundant_edges = lesser_edge['redundant_edges']
                    del lesser_edge['redundant_edges']
                    redundant_edges.append(lesser_edge)
                else:
                    #Old edge is bigger, keep it
                    g.edges[n1, n2]['redundant_edges'].append(edgeparams)
                    continue

            edgeparams['redundant_edges']=redundant_edges

            g.add_edge(n1, n2, **edgeparams)

            if redundant_edges: # sanity check for reference leaks
                assert g.edges[n1, n2]['channel_id'] != redundant_edges[0]['channel_id']

        return g

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

            return cls.fromjson()

        else:
            # fromconfig will create and exit if the config is missing
            ni = NodeInterface.fromconfig()

            # else load from lnd
            print('Fetching graph data from lnd')
            return cls.fromlnd(lndnode=ni)



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

