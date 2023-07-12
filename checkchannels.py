#!/usr/bin/env python3
"""
Connects to LND and checks channels for issues

"""
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import time
from math import ceil

from lib.nodeinterface import NodeInterface
from lib.lnGraph import lnGraph
from lib import fastcentrality

centralitycheckcount = 23
youththresholdweeks = 6


def printchannelflags(mynode):
    print('Chanid              kCap  RBal Alias                Flags')
    myneighbours = {}
    currentblockheight = mynode.GetInfo().block_height

    for chan in mynode.ListChannels().channels:

        flags = []
        chanid = chan.chan_id
        alias = mynode.getAlias(chan.remote_pubkey)
        kcap = chan.capacity / 1000
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

        chanblock = chanid >> 40  # Block when the channel was created
        chanage = currentblockheight - chanblock

        if chanage < 1008 * youththresholdweeks:
            weeks = ceil(chanage / 1008)
            flags.append(f'young<{weeks}w')

        if chan.initiator:
            if remote_ratio > 0.95: flags.append('depleted')
        else:
            if remote_ratio < 0.05: flags.append('depleted')

        print(chanid, f'{kcap:5.0f}{remote_ratio:6.1%} {alias[:20]:20}', *flags)
        myneighbours[chan.remote_pubkey] = {'flags': flags,
                                            'usage': totalsatsmoved,
                                            'age': chanage}

    return myneighbours


def newcentralityremoval(graphcopy, mynodekey, peer2remove):
    graphcopy.remove_edge(peer2remove, mynodekey)

    bc = fastcentrality.betweenness(graphcopy)

    ourcent = bc[mynodekey]
    theircent = bc[peer2remove]

    # undo for the next check in the batch
    graphcopy.add_edge(peer2remove, mynodekey)

    return ourcent, theircent


def printcentralitydiffs(mynode, myneighbours):
    peersbyusage = [i[0] for i in
                    sorted(myneighbours.items(), key=lambda n: n[1]['usage'])
                    if ('private' not in i[1]['flags']
                        and i[1]['age'] >= 1008 * youththresholdweeks)
                    ]
    underusedpeers = peersbyusage[:centralitycheckcount]

    if len(underusedpeers) == 0:
        print('All channels are young, skipping removal analysis')
        return

    print('\nFetching graph for further analysis')
    # Get the graph in networkx format, for reusing tools
    graph = lnGraph.fromlnd(lndnode=mynode, include_unannounced=True)
    mynodekey = mynode.GetInfo().identity_pubkey

    print(f'Checking up to {centralitycheckcount} least used public channels for removal centrality impact')
    peersbyremovalcentralityimpact = []
    with ProcessPoolExecutor() as executor:
        t = time.time()

        base_centrality_future = executor.submit(fastcentrality.betweenness, graph)

        results = executor.map(newcentralityremoval,
                               repeat(graph.copy()),
                               repeat(mynodekey),
                               underusedpeers)

        print('Waiting for base centrality computation to finish')
        print('An SSL warning may appear, please contact if you know the cause')
        base_centralities = base_centrality_future.result()
        mycurrentcentrality = base_centralities[mynodekey]
        counter = 0
        njobs = len(underusedpeers)
        print(f'Progress: {counter}/{njobs} {counter / njobs:.1%}',
              f'Elapsed time {time.time() - t:.1f}s',
              end='\r')

        for node_key, newcent in zip(underusedpeers, results):
            ournewcent, theirnewcent = newcent
            ourcentdelta = ournewcent - mycurrentcentrality
            theircentdelta = theirnewcent - base_centralities[node_key]

            peersbyremovalcentralityimpact.append((ourcentdelta, theircentdelta, node_key))

            counter += 1
            print(f'Progress: {counter}/{njobs} {counter / njobs:.1%}',
                  f'Elapsed time {time.time() - t:.1f}s',
                  end='\r')

    print(f'Completed centrality difference calculations in {time.time() - t:.1f}s')

    peersbyremovalcentralityimpact.sort(reverse=True)
    print('Centrality change if channel closed:')
    print('    Us   Them', 'Alias', 'Flags')
    for ourcentdelta, theircentdelta, pub_key in peersbyremovalcentralityimpact:

        if abs(base_centralities[pub_key]) > 0:
            theirpdelta = f'{theircentdelta / base_centralities[pub_key]:+6.1%}'
        else:
            theirpdelta = f'{"N/A":>6}'

        print(f'{ourcentdelta / mycurrentcentrality:+6.1%}',
              theirpdelta,
              f'{graph.nodes[pub_key]["alias"][:20]:20}',
              *myneighbours[pub_key]['flags']
              )


if __name__ == '__main__':
    mynode = NodeInterface.fromconfig()
    myneighbours = printchannelflags(mynode)

    printcentralitydiffs(mynode, myneighbours)
