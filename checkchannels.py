#!/usr/bin/env python3
"""
Connects to LND and checks channels for issues

"""
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import time

from nodeinterface import NodeInterface
import loadgraph
import fastcentrality

centralitycheckcount = 23

def printchannelflags(mynode):
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
    print('\nFetching graph for further analysis')
    # Get the graph in networkx format, for reusing tools
    graph = loadgraph.lnGraph.fromlnd(lndnode=mynode)
    mynodekey = mynode.GetInfo().identity_pubkey

    print(f'Checking up to {centralitycheckcount} least used public channels for removal centrality impact')
    peersbyusage = [i[0] for i in
                    sorted(myneighbours.items(), key=lambda n:n[1]['usage'])
                    if 'private' not in i[1]['flags']]
    underusedpeers = peersbyusage[:centralitycheckcount]

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
        print(f'Progress: {counter}/{njobs} {counter/njobs:.1%}',
          f'Elapsed time {time.time()-t:.1f}s',
          end='\r')

        for node_key, newcent in zip(underusedpeers, results):
            ournewcent, theirnewcent = newcent
            ourcentdelta = ournewcent - mycurrentcentrality
            theircentdelta = theirnewcent - base_centralities[node_key]

            peersbyremovalcentralityimpact.append((ourcentdelta, theircentdelta, node_key))

            counter += 1
            print(f'Progress: {counter}/{njobs} {counter/njobs:.1%}',
                  f'Elapsed time {time.time()-t:.1f}s',
                  end='\r')

    print(f'Completed centrality difference calculations in {time.time()-t:.1f}s')

    peersbyremovalcentralityimpact.sort(reverse=True)
    print('Centrality change if channel closed:')
    print('    Us   Them','Alias','Flags')
    for ourcentdelta, theircentdelta, pub_key in peersbyremovalcentralityimpact:

        print(f'{ourcentdelta/mycurrentcentrality:+6.1%}',
              f'{theircentdelta/base_centralities[pub_key]:+6.1%}',
              f'{graph.nodes[pub_key]["alias"][:20]:20}',
              *myneighbours[pub_key]['flags']
              )


if __name__ == '__main__':
    mynode = NodeInterface.fromconfig()
    myneighbours = printchannelflags(mynode)

    printcentralitydiffs(mynode, myneighbours)

