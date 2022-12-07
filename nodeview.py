#!/usr/bin/env python3
"""
Plots some node statistics for comparing with neighbours
This script requires describegraph.json

The blue dot represents the node being viewed relative to peers
"""
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from lnGraph import lnGraphV2

median_payment = 200e3  # sat, Payment size for calculating routing costs
# TODO: Should ignore channels smaller than (2x? 4-5x?) the above
feeceiling = 5000  # PPM, ignore fees higher than this

if len(sys.argv) > 1 and len(sys.argv[1]) <= 66:
    node2view = sys.argv[1]
else:
    print('Please enter the pubkey or alias of the node of interest')
    node2view = input('Pubkey: ')

print('Loading graph')
g = lnGraphV2.autoload()

if len(node2view) < 66:
    try:
        node2view = g.nodes.find(alias=node2view)['pub_key']
    except ValueError:
        print(f'Could not find node with alias "{node2view}"')
        exit()

node = g.nodes.find(node2view)

print('Viewing', node['alias'])

# Collect data for histogram
chanstats = dict(chansizes=[], peersizes=[], peerchancounts=[],
                 fees=dict(inrate=[], outrate=[], inbase=[], outbase=[]))

channel_data = g.channels.select(_from=node.index)
for chan in channel_data:
    assert node2view == chan['local_pubkey']
    assert isinstance(chan['capacity'], int)

    if chan['policy_in'] is None or chan['policy_out'] is None:
        continue  # Skip this channel

    chanstats['chansizes'].append(chan['capacity'])

    peerkey = chan['remote_pubkey']
    peer = g.nodes.find(pub_key=peerkey)
    chanstats['peersizes'].append(peer['capacity'])
    chanstats['peerchancounts'].append(peer['num_channels'])

    def appendfeedata(outrate, outbase, inrate, inbase):
        # Rates are in PPM, /1e6 to get a fraction
        # Base is in msat, /1e3 to get sat
        chanstats['fees']['outrate'].append(int(outrate))
        chanstats['fees']['outbase'].append(int(outbase))
        chanstats['fees']['inrate'].append(int(inrate))
        chanstats['fees']['inbase'].append(int(inbase))

    appendfeedata(chan['policy_out']['fee_rate_milli_msat'],
                  chan['policy_out']['fee_base_msat'],
                  chan['policy_in']['fee_rate_milli_msat'],
                  chan['policy_in']['fee_base_msat'])


# Convert to array
chanstats['chansizes'] = np.asarray(chanstats['chansizes'])
chanstats['peersizes'] = np.asarray(chanstats['peersizes'])
for k, v in chanstats['fees'].items():
    chanstats['fees'][k] = np.array(v)


def plothists():
    chansizebins = np.array([0, 2e6, 5e6, 10e6, 17e6, 20e6])

    if max(node['capacities']) > chansizebins[-1]:
        chansizebins[-1] = max(node['capacities'])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.tight_layout()
    ax1.hist(chanstats['chansizes'] / 1e6, align='right')
    ax1.set_title('Channel Size Distribution')
    ax1.set_xlabel('Channel size (Msat)')
    ax2.hist(chanstats['peersizes'] / 1e8, align='right')
    ax2.set_title('Peer Size Distribution')
    ax2.set_xlabel('Peer capacity (BTC)')

    median_in_fees = (chanstats['fees']['inbase'] / 1e3 + chanstats['fees']['inrate'] / 1e6 * median_payment)
    median_out_fees = (chanstats['fees']['outbase'] / 1e3 + chanstats['fees']['outrate'] / 1e6 * median_payment)

    ax3.hist(median_in_fees, align='right')
    ax4.hist(median_out_fees, align='right')
    ax3.set_title('Receiving Fee Distribution')
    ax4.set_title('Sending Fee Distribution')

    ax3.set_xlabel(f'Sats to route {median_payment / 1e3:.0f}ksat in')
    ax4.set_xlabel(f'Sats to route {median_payment / 1e3:.0f}ksat out')

    plt.show()


# ~ plothists()

def pltboxes():
    ncols = 5
    fig = plt.figure()

    # ~ fig.canvas.set_window_title('Viewing '+node['alias'])

    peerchanax = plt.subplot(1, ncols, 1)
    peersizeax = plt.subplot(1, ncols, 2)
    chansizeax = plt.subplot(1, ncols, 3)
    feeax = plt.subplot(1, ncols, (4, 5))

    feeax.set_title(f'Fee for {median_payment / 1e3:.0f}ksat Forward')

    # ~ print(list(zip(chanstats['peerchanco    unts'], [g.nodes[k]['alias'] for k in g.adj[node2view]])))
    peerchanax.boxplot(chanstats['peerchancounts'], showmeans=True)
    peerchanax.scatter([1], [node['num_channels']])
    peerchanax.set_title('Peer Channel Counts')
    peerchanax.set_ylabel('Peer Channel Count')

    peersizeax.boxplot(chanstats['peersizes'] / 1e8, showmeans=True)
    peersizeax.scatter([1], [node['capacity'] / 1e8])
    peersizeax.set_title('Peer Sizes')
    peersizeax.set_ylabel('Peer Capacity (BTC)')

    chansizeax.boxplot(chanstats['chansizes'] / 1e6, showmeans=True)
    chansizeax.set_title('Channel Sizes')
    chansizeax.set_ylabel('Channel Size (Msat)')

    median_in_fees = (chanstats['fees']['inbase'] / 1e3 + chanstats['fees']['inrate'] / 1e6 * median_payment)
    median_out_fees = (chanstats['fees']['outbase'] / 1e3 + chanstats['fees']['outrate'] / 1e6 * median_payment)

    median_in_fees_ppm = median_in_fees * 1e6/median_payment
    median_out_fees_ppm = median_out_fees * 1e6/median_payment

    # Remove outliers
    median_in_fees_ppm = median_in_fees_ppm[np.where(median_in_fees_ppm <= feeceiling)[0]]
    median_out_fees_ppm = median_out_fees_ppm[np.where(median_out_fees_ppm <= feeceiling)[0]]

    feeax.boxplot([median_in_fees_ppm, median_out_fees_ppm],
                  labels=['Receiving', 'Sending'], showmeans=True)
    feeax.set_ylabel('Fee (PPM)')

    fig.tight_layout()
    plt.show()


pltboxes()
