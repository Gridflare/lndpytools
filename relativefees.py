#!/usr/bin/env python3
"""
Check how our fees compare to other fees peers are charged
This script requires describegraph.json
"""
import sys

import numpy as np
import networkx as nx

import loadgraph

if len(sys.argv) > 1 and len(sys.argv[1]) == 66:
    mynodekey = sys.argv[1]
else:
    print('Please enter the pubkey of the node of interest')
    mynodekey = input('Pubkey: ')

#  Payment size for calculating routing costs
median_payment = 100e3 # sat

# ignore the fees on channels smaller than this
minchancapacity = median_payment*4

print('Loading graph')
g = loadgraph.lnGraph.autoload()
nx.freeze(g)

def getNodesChannels(pubkey):
    pass

inboundfees = {}
print('Checking peer fees')
for peerkey in g.adj[mynodekey]:
    inboundfees[peerkey] = []

    for peerpeerkey in g.adj[peerkey]:
        peerchan = g.edges[peerkey, peerpeerkey]

        commonchannels = [g.edges[peerkey, peerpeerkey]]
        commonchannels.extend(commonchannels[0]['redundant_edges'])

        thispairinfees = []
        for peerchan in commonchannels:

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

            thispairinfees.append(int(inboundbase)/1e3 + int(inboundrate)/1e6 * median_payment)

        if len(thispairinfees) > 0: # Can happen with outbound-only peers
            inboundfees[peerkey].append(np.median(thispairinfees))

print('Checking our fees')
myfeepercentiles = []
print('My_fee %ile Chan_cap_ksat Alias')
for peerkey in g.adj[mynodekey]:
    peer = g.nodes[peerkey]
    peerinfees = np.array(sorted(inboundfees[peerkey]))

    commonchannels = [g.edges[mynodekey, peerkey]]
    commonchannels.extend(commonchannels[0]['redundant_edges'])

    for chan in commonchannels:

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
        myfeep = np.sum(peerinfees < myfee) / len(peerinfees)
        myfeepercentiles.append(myfeep)

        print(f"{myfee:6.1f} {myfeep:5.1%} {chan['capacity']/1e3:5.0f}",peer['alias'])


print('\n STATS')
print(f'Max {max(myfeepercentiles):.1%}')
print(f'Min {min(myfeepercentiles):.1%}')
print(f'Median  {np.median(myfeepercentiles):.1%}')
print(f'Average {np.mean(myfeepercentiles):.1%}')





