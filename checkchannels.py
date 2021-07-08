#!/usr/bin/env python3
"""
Connects to LND and checks channels for issues
WIP
"""

from nodeinterface import NodeInterface


centralitycheckcount = 23

mynode = NodeInterface.fromconfig()

print('Chanid              kCap  RBal Alias                Flags')
myneighbours = {}
for chan in mynode.ListChannels().channels:

    flags = []
    chanid = chan.chan_id
    alias = mynode.getAlias(chan.remote_pubkey)
    kcap = chan.capacity/1000
    remote_ratio = chan.remote_balance / chan.capacity

    if chan.private:
        flags.append('private')

    if not chan.active:
        flags.append('inactive')

    totalsatsmoved = chan.total_satoshis_sent + chan.total_satoshis_received
    if totalsatsmoved == 0:
        flags.append('unused')
    elif chan.total_satoshis_sent == 0:
        flags.append('never_sent')
    elif chan.total_satoshis_received == 0:
        flags.append('never_rcvd')

    if chan.initiator:
        if remote_ratio > 0.95: flags.append('depleted')
    else:
        if remote_ratio < 0.05: flags.append('depleted')

    print(chanid, f'{kcap:5.0f}{remote_ratio:6.1%} {alias[:20]:20}', *flags)
    myneighbours[chan.remote_pubkey] = {'flags':flags, 'usage':totalsatsmoved}

print('\nFetching graph for further analysis')
graph = mynode.DescribeGraph()
print(len(graph.nodes), len(graph.edges))


print(f'Checking up to {centralitycheckcount} least used public channels for removal centrality impact')




