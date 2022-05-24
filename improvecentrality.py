#!/usr/bin/env python3
"""
This script requires describegraph.json and will generate its own config file.
Running on node hardware is not recommended, copy describegraph.json to a
desktop.

This script attempts to find peers that will substantially improve a node's
betweenness centrality. This does not guarantee that suggestions are good
routing nodes, do not blindly connect to the script's suggestions.

Initial candidate selection is done by information within the graph data,
further refinement is done using the 1ML availability metric and a score.

The score used for the preliminary ranking of candidates is a modified farness
metric. This score has no physical meaning, but appears to correlate well with
betweenness centrality, while being much easier to compute.

Since the end goal is improvement of centrality, and the score isn't perfect,
the script will compute how each potential peer will affect centrality.

The easiest speedup is to limit the number of nodes that pass final selection
to a number your CPU can reasonably process,
finalcandidatecount = (n*num_cpu_cores)-1 is my suggestion.

If you have an opinion on what the minimum size for a channel to be relevant to
the routing network is, you can dial that in, higher values will simplify the
graph more and improve performance.

"""

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import pandas as pd
import numpy as np
from scipy import stats

from GraphFilter import GraphFilter
from CandidateFilter import CandidateFilter
from lnGraph import lnGraph
from fastcentrality import *
from bc_utils import *
import time


def get_farness_score(peer2add, myfarness, graphcopy, map, mynodekey):
    # Modify the graph with a simulated channel
    graphcopy.add_edge(map[peer2add], map[mynodekey])

    mynewfarness = 1 / closeness(graphcopy, map[mynodekey])
    myfarnessdelta = mynewfarness - myfarness

    # Since this function is batched, and making a fresh copy is slow,
    # Make sure all changes are undone
    graphcopy.delete_edges([(map[peer2add], map[mynodekey])])

    # Want this data from the unmodified graph
    # Otherwise their score will be lowered if the channel
    # is too beneficial to them
    theirfarness = 1 / closeness(graphcopy, map[peer2add])

    # This is where the magic happens
    # Nodes that reduce our farness,
    # as well as nodes with a high farness,
    # are prioritized. But especially nodes that have both.
    farnessscore = np.sqrt(abs(myfarnessdelta)) + theirfarness / 1000

    return farnessscore


def calculate_farness_scores(candidatekeys, graph, map, mynodekey):

    print('Running modified farness score calculations')

    t = time.time()
    farnesscores = {}
    myfarness = 1 / closeness(graph, map[mynodekey])

    with ProcessPoolExecutor() as executor:
        scoreresults = executor.map(get_farness_score,
                                    candidatekeys,
                                    repeat(myfarness),
                                    repeat(graph.copy()),
                                    repeat(map),
                                    repeat(mynodekey),
                                    chunksize=128)

    for nkey, score in zip(candidatekeys, scoreresults):
        farnesscores[nkey] = score

    print(f'Completed modified farness score calculations in {time.time() - t:.1f}s')

    return farnesscores


def get_new_centrality(peer2add, graphcopy, map, mynodekey):

    graphcopy.add_edge(map[peer2add], map[mynodekey])

    newbc = betweenness(graphcopy, map[mynodekey])

    # Remove in case the same instance is reused due to batching
    graphcopy.delete_edges([(map[peer2add], map[mynodekey])])

    return newbc


def calculate_centrality_deltas(candidatekeys, graph, map, mynodekey):

    t = time.time()
    centralitydeltas = {}

    with ProcessPoolExecutor() as executor:
        print('Starting baseline centrality computation')
        mycentralityfuture = executor.submit(betweenness, graph, map[mynodekey])

        print('Queuing computations for new centralities')

        newcentralities = executor.map(get_new_centrality,
                                       candidatekeys,
                                       repeat(graph.copy()),
                                       repeat(map),
                                       repeat(mynodekey),
                                       chunksize=4)

        print('Waiting for baseline centrality calculation to complete')
        myoldcentrality = mycentralityfuture.result()

        print('Our current centrality is approximately', int(myoldcentrality))

        print('Collecting centrality results, this may take a while')

        counter = 0
        njobs = len(candidatekeys)
        print(f'Progress: {counter}/{njobs} {counter / njobs:.1%}',
              f'Elapsed time {time.time() - t:.1f}s',
              end='\r')

        for nkey, newcentrality in zip(candidatekeys, newcentralities):
            centralitydelta = newcentrality - myoldcentrality
            centralitydeltas[nkey] = centralitydelta

            counter += 1
            print(f'Progress: {counter}/{njobs} {counter / njobs:.1%}',
                  f'Elapsed time {time.time() - t:.1f}s',
                  end='\r')

    print(f'Completed centrality difference calculations in {time.time() - t:.1f}s')
    return centralitydeltas, myoldcentrality

def safe_div(x,y):
    if y==0: return 0
    return x/y

def print_results(centralitydeltas, mycurrentcentrality, filtered_graph, farness_scores, validate):
    cols = 'delta', 'MFscor', 'Avail', 'Relbty', 'Alias', 'Pubkey'
    print(*cols)
    export_dict = {k: [] for k in cols}
    cdeltascores = []
    mfscores = []

    for nkey, cdelta in sorted(centralitydeltas.items(), key=lambda i: -i[1]):
        nodedata = filtered_graph.nodes[nkey]
        alias = nodedata['alias']
        mfscore = farness_scores[nkey]
        arank = get_1ml_stats(nkey)['noderank']['availability']
        reliability = 1 - nodedata['disabledcount']['receiving'] / filtered_graph.degree(nkey)

        cdeltascores.append(cdelta)
        mfscores.append(mfscore)

        cdeltastr = f'{safe_div(cdelta, mycurrentcentrality):6.1%}'
        relbtystr = f'{reliability:6.1%}'
        export_dict['delta'].append(cdeltastr)
        export_dict['MFscor'].append(mfscore)
        export_dict['Avail'].append(arank)
        export_dict['Relbty'].append(relbtystr)
        export_dict['Alias'].append(alias)
        export_dict['Pubkey'].append(nkey)

        # print(f'{cdelta / mycurrentcentrality:+6.1%} {mfscore:6.2f} {arank:5}, {relbtystr}, {alias:>50}, {nkey:>20}')

    if validate:
        reg = stats.linregress(cdeltascores, mfscores)
        r = round(reg.rvalue, 3)
        print('Heuristic validation found an r value of', r)
        if r < 0.7:
            print('r is low, you will need a higher finalcandidatecount to compensate')
        elif r < 0.8:
            print('r is on the low end of the expected range for this heuristic')
            print('consider a higher finalcandidatecount to compensate')
        elif r < 0.9:
            print('r is on the high end of the expected range for this heuristic')
        else:
            print('r is better than expected for this heuristic')
    export_pd = pd.DataFrame(export_dict)
    print(export_pd.to_markdown())
    return export_pd


def save_recommendations(export_dict, config):
    csv_export_name = config['Other'].get('csvexportname')
    if csv_export_name:
        df = pd.DataFrame(export_dict)
        df.set_index('Pubkey')
        df.to_csv(csv_export_name)


def main():
    parser = make_parser()
    args = parser.parse_args()
    config = load_config(args.conffile)
    pub_key = config['Node']['pub_key']

    graph_filters = config['GraphFilters']
    graph = lnGraph.autoload(expirehours=False)

    filtered_graph = GraphFilter(graph, pub_key, graph_filters).filtered_g
    fast_graph, map = nx2ig(filtered_graph)

    if pub_key not in filtered_graph.nodes:
        print(f'Failed to find a match for pub_key={pub_key} in the graph')
        print('Please double check improvecentrality.conf')
        exit()
    if filtered_graph.degree(pub_key) < 2:
        print('This script requires your node to have a minimum of 2 stable, public channels')
        print('Your node does not meet this requirement at this time.')
        exit()

    print('Performing analysis for', filtered_graph.nodes[pub_key]['alias'])
    candidate_filters = config['CandidateFilters']
    candidate_filters["pub_key"] = pub_key
    if args.validate:
        candidate_filters['finalcandidatecount'] = '400'
    channel_candidates = CandidateFilter(filtered_graph, candidate_filters).filtered_candidates
    print('First filtering pass found', len(channel_candidates), 'candidates for new channels')

    farness_scores = calculate_farness_scores(channel_candidates, fast_graph, map, pub_key)
    candidates_by_farness = sorted(channel_candidates, key=lambda k: -farness_scores[k])

    max_availability = candidate_filters.getint('max1mlavailability')
    final_candidate_count = candidate_filters.getint('finalcandidatecount')
    final_candidates = select_by_1ml(candidates_by_farness, max_availability,
                                     final_candidate_count)
    
    if len(final_candidates) == 0:
        print('No candidates found, is your graph stale?')
        print('If issue persists, delete describegraph.json and improvecentrality.conf')
        raise ValueError('No valid candidates')

    centrality_deltas, current_centrality = calculate_centrality_deltas(
        final_candidates, fast_graph, map, pub_key)
    export_dict = print_results(centrality_deltas, current_centrality, filtered_graph, farness_scores, args.validate)
    save_recommendations(export_dict, config)


if __name__ == '__main__':
    main()
