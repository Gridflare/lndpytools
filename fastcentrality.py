#!/usr/bin/env python3
"""
Fucntions for using igraph tools on networkx graphs

"""
import networkx as nx
import igraph



def nx2ig(nxgraph):
    """
    Convert networkx graph to igraph
    For centrality we only care about keys and channels
    igraph uses integer indexing, so a map will also be created.
    """

    nxnodekeys = nxgraph.nodes
    nxnodemap = {nk:i for i, nk in enumerate(nxnodekeys)}

    ig = igraph.Graph()
    ig.add_vertices(len(nxnodekeys))

    igedges = [(nxnodemap[nk1], nxnodemap[nk2])
               for nk1, nk2 in nxgraph.edges]

    ig.add_edges(igedges)

    return ig, nxnodemap


def betweenness(graph, nodeid=None):
    remap2nx = False
    if isinstance(graph, nx.Graph):
        # Convert nx graph to igraph format
        graph, nxnodemap = nx2ig(graph)
        # Convert nx node id to igraph integer
        if nodeid is not None:
            nodeid = nxnodemap[nodeid]
        else:
            remap2nx = True

    bc = graph.betweenness(nodeid)
    if remap2nx:
        # Assumes nxnodemap dict has keys in order
        bc = {nk:c for nk, c in zip(nxnodemap.keys(), bc)}

    return bc

def closeness(graph, nodeid=None):
    remap2nx = False
    if isinstance(graph, nx.Graph):
        # Convert nx graph to igraph format
        graph, nxnodemap = nx2ig(graph)
        # Convert nx node id to igraph integer
        if nodeid is not None:
            nodeid = nxnodemap[nodeid]
        else:
            remap2nx = True

    cc = graph.closeness(nodeid, normalized=False)
    if remap2nx:
        # Assumes nxnodemap dict has keys in order
        cc = {nk:c for nk, c in zip(nxnodemap.keys(), cc)}

    return cc


if __name__ == '__main__':
    # just a quick test
    nxg = nx.Graph()
    nxg.add_nodes_from(['A', 'B', 'C', 'D','E'])
    nxg.add_edges_from([('A','B'),('A','C'),('A','E'),
                        ('C','D'), ('D','E')])

    ig, nxnodemap = nx2ig(nxg)

    print(nxnodemap)
    print(ig)

    import time

    t = time.time()
    igbc = ig.betweenness()
    print('IG BCentrality in (ms)', (time.time()-t)*1000)
    print(igbc)

    t = time.time()
    igcc = ig.closeness()
    print('IG CCentrality in (ms)', (time.time()-t)*1000)
    print(igcc)

    t = time.time()
    nxbc = nx.algorithms.centrality.betweenness_centrality(nxg, normalized=False)
    print('NX BCentrality in (ms)', (time.time()-t)*1000)
    print(nxbc)


