
import json

import networkx as nx

# Multigraph is the most honest representation,
# but in practice we only care about the largest channels
# Some algorithms also prefer the simpler graph


def fromjson():
    with open('describegraph.json') as f:
        graphdata = json.load(f)

    g = nx.Graph()
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

        if g.has_edge(n1, n2):
            # Need to decide to overwrite or skip
            if g.edges[n1, n2]['capacity'] > cap:
                # Keep the old edge if it is bigger
                continue

        g.add_edge(n1, n2,
                   channel_id=edge['channel_id'],
                   chan_point=edge['chan_point'],
                   last_update=edge['last_update'],
                   capacity=cap,
                   node1_pub=n1,
                   node2_pub=n2,
                   node1_policy=edge['node1_policy'],
                   node2_policy=edge['node2_policy'],
                   )

    return g
