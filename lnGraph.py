import json
import os
import time

from networkx import Graph as nxGraph
from igraph import Graph as igGraph
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

    @classmethod
    def fromjson(cls, graphfile='describegraph.json'):
        raise NotImplementedError

    @classmethod
    def fromlnd(cls, lndnode: NodeInterface=None, include_unannounced=False):
        raise NotImplementedError

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


class lnGraph(lnGraphBase, nxGraph):
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

class lnGraphBaseIGraph(lnGraphBase, igGraph):
    @property
    def nodes(self):
        return self.vs

    @property
    def channels(self):
        return self.es

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_channels(self):
        # Divide by 2, each channel is represented by 2 directed edges
        return len(self.channels)//2


class lnGraphV2(lnGraphBaseIGraph):
    @staticmethod
    def _init_temp_structure():
        # temporary values used during initialization
        v = dict(
            nodeattrs = {a:[] for a in ['pub_key', 'last_update', 'alias', 'color', 'addresses']},
            edgeattrs = {a:[] for a in ['channel_id', 'chan_point', 'capacity', 'last_update']},
            policies1 = [],
            policies2 = [],
            edgeids1 = [],
            edgeids2 = [],
            nodepubs1 = [],
            nodepubs2 = [],
        )

        return v

    @staticmethod
    def _init_record_channel(init_vars, nodeindexmap, n1pub, n2pub,
                             chan_cap, policy1, policy2):
        # Record the channel to records
        init_vars['edgeids1'].append((n1pub,n2pub))
        init_vars['edgeids2'].append((n2pub,n1pub))
        init_vars['nodepubs1'].append(n1pub)
        init_vars['nodepubs2'].append(n2pub)

        n1i = nodeindexmap[n1pub]
        n2i = nodeindexmap[n2pub]

        init_vars['nodeattrs']['capacity'][n1i] += chan_cap
        init_vars['nodeattrs']['capacity'][n2i] += chan_cap
        init_vars['nodeattrs']['num_channels'][n1i] += 1
        init_vars['nodeattrs']['num_channels'][n2i] += 1

        init_vars['policies1'].append(policy1)
        init_vars['policies2'].append(policy2)

    @staticmethod
    def _init_setflattenedpolicies(policies:list,direction:str,dest:dict):
        policy_keys = ['time_lock_delta', 'min_htlc', 'max_htlc_msat',
                       'fee_base_msat', 'fee_rate_milli_msat',
                       'disabled', 'last_update']

        for k in policy_keys:
            dest[f'{k}_{direction}'] = [
                None if p is None else p[k] for p in policies
            ]

    @classmethod
    def _init_setpolicies1(cls, init_vars):
        # Set policies from the perspective of node 1
        init_vars['edgeattrs']['local_pubkey'] = init_vars['nodepubs1']
        init_vars['edgeattrs']['remote_pubkey'] = init_vars['nodepubs2']

        cls._init_setflattenedpolicies(
            init_vars['policies1'],'out',init_vars['edgeattrs'])
        cls._init_setflattenedpolicies(
            init_vars['policies2'],'in',init_vars['edgeattrs'])

    @classmethod
    def _init_setpolicies2(cls, init_vars):
        # Set policies from the perspective of node 2
        init_vars['edgeattrs']['local_pubkey'] = init_vars['nodepubs2']
        init_vars['edgeattrs']['remote_pubkey'] = init_vars['nodepubs1']

        cls._init_setflattenedpolicies(
            init_vars['policies2'],'out',init_vars['edgeattrs'])
        cls._init_setflattenedpolicies(
            init_vars['policies1'],'in',init_vars['edgeattrs'])

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

            data_keys = data.keys()
            for k in intattrs:
                if k in data_keys:
                    data[k] = int(data[k])

            return data

        graph_elements = cls._init_temp_structure()

        for node in graphdata['nodes']:
            node = fixstr2int(node)
            for k in graph_elements['nodeattrs'].keys():
                graph_elements['nodeattrs'][k].append(node[k])

        n = len(graph_elements['nodeattrs']['pub_key'])
        graph_elements['nodeattrs']['capacity'] = [0]*n  # Increment while iterating channels
        graph_elements['nodeattrs']['num_channels'] = [0]*n  # Increment while iterating channels

        nodeindexmap = {k:i for i, k in
            enumerate(graph_elements['nodeattrs']['pub_key'])}

        for edge in graphdata['edges']:
            edge = fixstr2int(edge)
            n1pub = edge['node1_pub']
            n2pub = edge['node2_pub']
            cap = edge['capacity']

            for k in graph_elements['edgeattrs'].keys():
                graph_elements['edgeattrs'][k].append(edge[k])

            cls._init_record_channel(graph_elements, nodeindexmap,
                                    n1pub, n2pub, cap,
                                    fixstr2int(edge['node1_policy']),
                                    fixstr2int(edge['node2_policy']),
                                    )

        g.add_vertices(graph_elements['nodeattrs']['pub_key'],
                       graph_elements['nodeattrs'])

        # Using a directed graph, have to add edges twice because channels are bidirectionsl
        cls._init_setpolicies1(graph_elements)

        g.add_edges(graph_elements['edgeids1'], graph_elements['edgeattrs'])

        cls._init_setpolicies2(graph_elements)

        g.add_edges(graph_elements['edgeids2'], graph_elements['edgeattrs'])
        return g

    @classmethod
    def fromlnd(cls, lndnode: NodeInterface = None, include_unannounced=False):
        if lndnode is None:
            lndnode = NodeInterface()  # use defaults

        graphdata = lndnode.DescribeGraph(include_unannounced=include_unannounced)

        g = cls(directed=True)

        graph_elements = cls._init_temp_structure()

        for node in graphdata.nodes:
            for k in graph_elements['nodeattrs'].keys():
                a = getattr(node, k)
                if k == 'addresses':
                    a = gRPCadapters.addrs2dict(a)
                graph_elements['nodeattrs'][k].append(a)

        n = len(graph_elements['nodeattrs']['pub_key']) # Number of nodes in graph
        graph_elements['nodeattrs']['capacity'] = [0]*n  # Increment while iterating channels
        graph_elements['nodeattrs']['num_channels'] = [0]*n  # Increment while iterating channels

        nodeindexmap = {k:i for i, k in
                        enumerate(graph_elements['nodeattrs']['pub_key'])}

        for edge in graphdata.edges:
            n1pub = edge.node1_pub
            n2pub = edge.node2_pub
            cap = int(edge.capacity)

            for k in graph_elements['edgeattrs'].keys():
                graph_elements['edgeattrs'][k].append(getattr(edge, k))

            cls._init_record_channel(graph_elements, nodeindexmap,
                        n1pub,n2pub, cap,
                        gRPCadapters.nodepolicy2dict(edge.node1_policy),
                        gRPCadapters.nodepolicy2dict(edge.node2_policy),
                        )

        g.add_vertices(graph_elements['nodeattrs']['pub_key'],
                       graph_elements['nodeattrs'])

        # Using a directed graph, have to add edges twice to keep track of policies
        cls._init_setpolicies1(graph_elements)

        g.add_edges(graph_elements['edgeids1'], graph_elements['edgeattrs'])

        cls._init_setpolicies2(graph_elements)

        g.add_edges(graph_elements['edgeids2'], graph_elements['edgeattrs'])
        return g

    def simple(self):
        """
        Returns a copy without parallel edges,
        intelligently consolidates policy information
        """
        c = self.copy()

        unknown_in = c.es.select(disabled_in_eq=None)
        unknown_out = c.es.select(disabled_out_eq=None)

        pkeys = ['time_lock_delta', 'min_htlc', 'max_htlc_msat',
             'fee_base_msat', 'fee_rate_milli_msat',
             'disabled', 'last_update']

        combine_edges_method = {
            'capacity':'sum',
            'last_update': 'max',
            'local_pubkey': 'first',
            'remote_pubkey': 'first',
            'channel_id': 'first',
        }
        for d in ['in','out']:
            policies = {
                f'time_lock_delta_{d}': 'max',
                f'min_htlc_{d}': 'min',
                f'max_htlc_msat_{d}': 'max',
                f'fee_base_msat_{d}': 'max',
                f'fee_rate_milli_msat_{d}': 'max',
                f'disabled_{d}': all,
                f'last_update_{d}': 'max',
            }
            combine_edges_method.update(policies)

        # Can't use comparisons when a field is None
        for pk in pkeys:
            # -1 should be a safe placeholder
            unknown_in.set_attribute_values(f'{pk}_in', -1)
            unknown_out.set_attribute_values(f'{pk}_out', -1)

        s = c.simplify(combine_edges=combine_edges_method)

        # Reset unknown fields to None
        unknown_in = s.es.select(time_lock_delta_in_eq=-1)
        unknown_out = s.es.select(time_lock_delta_out_eq=-1)
        for pk in pkeys:
            unknown_in.set_attribute_values(f'{pk}_in', None)
            unknown_out.set_attribute_values(f'{pk}_out', None)

        return s

    def simple_nopolicy(self):
        """Faster than .simple(), discards policy info"""
        combine_edges_method = {
            'capacity': 'sum',
            'last_update': 'max',
            'local_pubkey': 'first',
            'remote_pubkey': 'first',
            'channel_id': 'first', # Keep, because useful as an index
        }
        return self.copy().simplify(combine_edges=combine_edges_method)

    def simple_nopolicy_undirected(self):
        """Faster than .simple(), discards policy info"""
        combine_edges_method = {
            'capacity': 'sum',
            'last_update': 'max',
            'channel_id': 'first',
        }
        g = self.copy()
        g.to_undirected(mode='collapse', combine_edges=combine_edges_method)
        return g

if __name__ == '__main__':
    print('Loading graph')
    g = lnGraphV2.autoload()
    # ~ g = lnGraphV2.fromlnd(lndnode=NodeInterface.fromconfig())
    print(g.summary())
    # find() grabs the first match, for testing
    n = g.nodes.find(alias='Gridflare')
    print(n)
    print(g.channels.find(_from=n))
