#!/usr/bin/env python3
"""
Check how our fees compare to other fees peers are charged
"""

import numpy as np
import networkx as nx

import loadgraph

mynodekey = '02e2bf9e87c7ba0ea046882d9b3301ca9a3b049aa920dccbd836c4008eac32d4e5' # Gridflare

median_payment = 100e3 # sat, Payment size for calculating routing costs

minchancapacity = median_payment*4

print('Loading graph')
g = loadgraph.fromjson()
nx.freeze(g)

inboundfees = {}
print('Checking peer fees')
for peerkey in g.adj[mynodekey]:
    inboundfees[peerkey] = []

    for peerpeerkey in g.adj[peerkey]:
        peerchan = g.edges[peerkey, peerpeerkey]

        if any((peerchan['capacity'] < minchancapacity,
                peerchan['node1_policy'] is None,
                peerchan['node2_policy'] is None,
            )):
            continue # ignore this insignificant channel

        if peerpeerkey == peerchan['node1_pub']:
            inboundrate = peerchan['node1_policy']['fee_rate_milli_msat']
            inboundbase = peerchan['node1_policy']['fee_base_msat']
        elif peerpeerkey == peerchan['node2_pub']:
            inboundrate = peerchan['node2_policy']['fee_rate_milli_msat']
            inboundbase = peerchan['node2_policy']['fee_base_msat']
        else:
            assert False

        infee = (int(inboundbase)/1e3 + int(inboundrate)/1e6 * median_payment)
        inboundfees[peerkey].append(infee)

print('Checking our fees')
myfeepercentiles = []
print('My_fee %ile Next_mult Chan_cap_ksat Alias')
for peerkey in g.adj[mynodekey]:
    peer = g.nodes[peerkey]
    peerinfees = np.array(sorted(inboundfees[peerkey]))
    chan = g.edges[mynodekey, peerkey]

    if chan['capacity'] < minchancapacity:
        continue # can't check a channel we skipped earlier

    if mynodekey == chan['node1_pub']:
        myfeerate = chan['node1_policy']['fee_rate_milli_msat']
        myfeebase = chan['node1_policy']['fee_base_msat']
    elif mynodekey == chan['node2_pub']:
        myfeerate = chan['node2_policy']['fee_rate_milli_msat']
        myfeebase = chan['node2_policy']['fee_base_msat']
    else:
        assert False

    myfee = (int(myfeebase)/1e3 + int(myfeerate)/1e6 * median_payment)
    myfeerank = np.where(peerinfees==myfee)[0][0]+1 # higher is more expensive
    myfeep = myfeerank/len(peerinfees)
    myfeepercentiles.append(myfeep)

    # Ratio of our fee to a peer vs the next lowest fee to that peer
    nextlowestratio = myfee/peerinfees[myfeerank-2]

    print(f"{myfee:6.1f} {myfeep:4.0%} {nextlowestratio:4.1f} {chan['capacity']/1e3:5.0f}",peer['alias'])
    # ~ print(myfee, peerinfees[-7:])
    # ~ print(len(peerinfees), np.where(peerinfees==myfee)[0][0])

print('\n STATS')
print(f'Max {max(myfeepercentiles):.1%}')
print(f'Min {min(myfeepercentiles):.1%}')
print(f'Median  {np.median(myfeepercentiles):.1%}')
print(f'Average {np.mean(myfeepercentiles):.1%}')





