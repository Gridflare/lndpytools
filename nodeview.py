"""
Plots some node statistics for comparing with neighbours
"""


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import loadgraph

node2view = '02e2bf9e87c7ba0ea046882d9b3301ca9a3b049aa920dccbd836c4008eac32d4e5' # Gridflare
# ~ node2view = '03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f' # ACINQ
# ~ node2view = '02bb10aaa77a95a358cebb2d112c4de00e47c08f56e89b1acb4487ddd44cc98d6d' # microlancer
# ~ node2view = '02a04446caa81636d60d63b066f2814cbd3a6b5c258e3172cbdded7a16e2cfff4c' # bitstamp
# ~ node2view = '02816caed43171d3c9854e3b0ab2cf0c42be086ff1bd4005acc2a5f7db70d83774' # Fold
# ~ node2view = '033d8656219478701227199cbd6f670335c8d408a92ae88b962c49d4dc0e83e025' # bfx-lnd0
# ~ node2view = '021c97a90a411ff2b10dc2a8e32de2f29d2fa49d41bfbb52bd416e460db0747d0d' # LOOP

median_payment = 100e3 # sat, Payment size for calculating routing costs
# Should ignore channels smaller than (2x? 4-5x?) the above
feeceiling = 6000 #sats, ignore fees charge more than this, unidirectional channel?

g = loadgraph.fromjson()
# ~ print(list(g.nodes.keys())[0])
nx.freeze(g)


node = g.nodes[node2view]

print('Viewing', node['alias'])

# Collect data for histogram
chanstats = dict(chansizes=[], peersizes=[],peerchancounts=[],
                 fees=dict(inrate=[],outrate=[],inbase=[],outbase=[]))
for peerkey in g.adj[node2view]:
    chan = g.edges[node2view, peerkey]
    if chan['node1_policy'] is None or chan['node2_policy'] is None:
        continue # Skip this channel

    chanstats['chansizes'].append(chan['capacity'])
    chanstats['peersizes'].append(g.nodes[peerkey]['capacity'])
    chanstats['peerchancounts'].append(g.nodes[peerkey]['num_channels'])

    def appendfeedata(outrate, outbase, inrate, inbase):
        # Rates are in PPM, /1e6 to get a fraction
        # Base is in msat, /1e3 to get sat
        chanstats['fees']['outrate'].append(int(outrate))
        chanstats['fees']['outbase'].append(int(outbase))
        chanstats['fees']['inrate'].append(int(inrate))
        chanstats['fees']['inbase'].append(int(inbase))

    if node2view == chan['node1_pub']:
        appendfeedata(chan['node1_policy']['fee_rate_milli_msat'],
                      chan['node1_policy']['fee_base_msat'],
                      chan['node2_policy']['fee_rate_milli_msat'],
                      chan['node2_policy']['fee_base_msat'])

    elif node2view == chan['node2_pub']:
        appendfeedata(chan['node2_policy']['fee_rate_milli_msat'],
                      chan['node2_policy']['fee_base_msat'],
                      chan['node1_policy']['fee_rate_milli_msat'],
                      chan['node1_policy']['fee_base_msat'])

    else:
        assert False

# Convert to array
chanstats['chansizes'] = np.array(chanstats['chansizes'])
chanstats['peersizes'] = np.array(chanstats['peersizes'])
for k, v in chanstats['fees'].items():
    chanstats['fees'][k] = np.array(v)

def plothists():
    chansizebins = np.array([0, 2e6, 5e6, 10e6, 17e6, 20e6])

    if max(node['capacities']) > chansizebins[-1]:
        chansizebins[-1] = max(node['capacities'])

    fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2,2)
    fig.tight_layout()
    ax1.hist(chanstats['chansizes']/1e6, align='right')
    ax1.set_title('Channel Size Distribution')
    ax1.set_xlabel('Channel size (Msat)')
    ax2.hist(chanstats['peersizes']/1e8, align='right')
    ax2.set_title('Peer Size Distribution')
    ax2.set_xlabel('Peer capacity (BTC)')

    median_in_fees = (chanstats['fees']['inbase']/1e3 + chanstats['fees']['inrate']/1e6 * median_payment)
    median_out_fees = (chanstats['fees']['outbase']/1e3 + chanstats['fees']['outrate']/1e6 * median_payment)

    ax3.hist(median_in_fees, align='right')
    ax4.hist(median_out_fees, align='right')
    ax3.set_title('Receiving Fee Distribution')
    ax4.set_title('Sending Fee Distribution')

    ax3.set_xlabel(f'Sats to route {median_payment/1e3:.0f}ksat in')
    ax4.set_xlabel(f'Sats to route {median_payment/1e3:.0f}ksat out')

    plt.show()

# ~ plothists()

def pltboxes():
    ncols = 5
    fig = plt.figure()

    # ~ fig.canvas.set_window_title('Viewing '+node['alias'])

    peerchanax = plt.subplot(1, ncols, 1)
    peersizeax = plt.subplot(1, ncols, 2)
    chansizeax = plt.subplot(1, ncols, 3)
    feeax = plt.subplot(1, ncols, (4,5))

    feeax.set_title(f'Fee for {median_payment/1e3:.0f}ksat Forward')

    # ~ print(list(zip(chanstats['peerchanco    unts'], [g.nodes[k]['alias'] for k in g.adj[node2view]])))
    peerchanax.boxplot(chanstats['peerchancounts'], showmeans=True)
    peerchanax.scatter([1], [node['num_channels']])
    peerchanax.set_title('Peer Channel Counts')
    peerchanax.set_ylabel('Peer Channel Count')

    peersizeax.boxplot(chanstats['peersizes']/1e8, showmeans=True)
    peersizeax.scatter([1], [node['capacity']/1e8])
    peersizeax.set_title('Peer Sizes')
    peersizeax.set_ylabel('Peer Capacity (BTC)')

    chansizeax.boxplot(chanstats['chansizes']/1e6, showmeans=True)
    chansizeax.set_title('Channel Sizes')
    chansizeax.set_ylabel('Channel Size (Msat)')

    median_in_fees = (chanstats['fees']['inbase']/1e3 + chanstats['fees']['inrate']/1e6 * median_payment)
    median_out_fees = (chanstats['fees']['outbase']/1e3 + chanstats['fees']['outrate']/1e6 * median_payment)

    # Remove outliers
    median_in_fees = median_in_fees[np.where(median_in_fees<=feeceiling)[0]]
    median_out_fees = median_out_fees[np.where(median_out_fees<=feeceiling)[0]]

    feeax.boxplot([median_in_fees, median_out_fees],
                  labels=['Receiving','Sending'], showmeans=True)
    feeax.set_ylabel('Fee (sat)')

    fig.tight_layout()
    plt.show()


pltboxes()
