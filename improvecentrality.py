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

import json
import time
from concurrent.futures import ProcessPoolExecutor
import os
from itertools import repeat
import configparser

import requests
import networkx as nx
from networkx.algorithms import centrality
from networkx.algorithms.shortest_paths.unweighted import single_source_shortest_path_length
import pandas as pd
import numpy as np

import loadgraph
import fastcentrality

## Configuration starts here

# Node pubkeys that I plan to open a channel to, count these as already formed.
addchannels = [

            ]

# Assumed size of the new channels, should not currently have an effect
newchannelsize = 2e6

# Node pubkeys that I plan to close all channels to, count these as nonexistant.
removechannels = [

            ]

def loadconfig(conffile = 'improvecentrality.conf'):

    if os.path.isfile(conffile):
        config = configparser.ConfigParser()
        config.read(conffile)
        return config

    print('Config not found, will create', conffile)
    config = configparser.ConfigParser()
    config['Node'] = {'pub_key':'yournodepubkeyhere'}
    config['GraphFilters'] = {
                         # Ignore channels smaller than this during analysis,
                         # unless they connect to us. Higher values improve
                         # script performance and odds of good routes.
                         # However lower values give numbers closer to reality
                        'minrelevantchan':500_000,
                              }
    config['CandidateFilters'] ={
                        'minchancount': 8,
                        'maxchancount': 10000,
                        'mincapacitybtc': 0.2,
                        'maxcapacitybtc': 1000,
                        'minavgchan': 750_000,
                        # Default 4+ >1M channels, 2+ >2M channels
                        'minchannels':'1500k4 3M2',
                        'minreliability': 0.95,
                        # Node must be ranked better than this for availability on 1ml
                        'max1mlavailability': 1500,
                        # Limit the number of nodes passed to the final (slow!) centrality computation
                        'finalcandidatecount': 11,
                        }
    config['Other'] = {
                       # Export results to this CSV file. Set to None or '' to disable
                       'csvexportname':'newchannels.csv',
                       }

    with open(conffile, 'w') as cf:
        config.write(cf)

    print('Please complete the config and rerun')
    exit()

def validatenodechancapacities(chancaps, minchanstiers):
    """
    This function allows more granular selection of candidates than total or
    average capacity.
    """
    for tierfilter in minchanstiers:
        if 'k' in tierfilter:
            ksize, mincount = tierfilter.split('k')
            size = int(ksize)*1e3
        elif 'M' in tierfilter:
            Msize, mincount = tierfilter.split('M')
            size = int(Msize)*1e6
        else:
            raise RuntimeError('No recognized seperator in minchannel filter')

        if sum((c >= size for c in chancaps)) < int(mincount):
            return False

    return True

def determinereliability(candidatekey, graph):
    # Record reliability-related data into the node record

    node = graph.nodes[candidatekey]

   # This should never happen, but there is evidence that it does
    if graph.degree(candidatekey) == 0:
        print(node)
        raise RuntimeError(f"{node['alias']} has no valid channels")

    chankeypairs = graph.edges(candidatekey)
    node['disabledcount'] = {'sending':0, 'receiving':0}

    for chankeypair in chankeypairs:
        chan = graph.edges[chankeypair]

        if None in [chan['node1_policy'], chan['node2_policy']]:
            continue # TODO: Might want to add a penalty to this

        if candidatekey == chan['node1_pub']:
            if chan['node1_policy']['disabled']:
                node['disabledcount']['sending'] += 1
            if chan['node2_policy']['disabled']:
                node['disabledcount']['receiving'] += 1
        elif candidatekey == chan['node2_pub']:
            if chan['node2_policy']['disabled']:
                node['disabledcount']['sending'] += 1
            if chan['node1_policy']['disabled']:
                node['disabledcount']['receiving'] += 1
        else:
            assert False

    return 1 - node['disabledcount']['receiving'] / graph.degree(candidatekey)

def selectinitialcandidates(graph, filters):
    # Conditions a node must meet to be considered for further connection analysis
    minchancount = filters.getint('minchancount')
    mincapacity = int(filters.getfloat('mincapacitybtc')*1e8)
    maxchancount = filters.getint('maxchancount')
    maxcapacity = int(filters.getfloat('maxcapacitybtc')*1e8)
    minavgchan = filters.getint('minavgchan')
    minreliability = filters.getfloat('minreliability')
    minchannelsstr = filters['minchannels']
    minchanstiers = minchannelsstr.split()

    def filtercandidatenodes(n, graph, filters):
        # This is the inital filtering pass, it does not include 1ML or centrality

        # If already connected or is ourself, abort
        nkey = n['pub_key']
        if graph.has_edge(mynodekey, nkey) or mynodekey == nkey:
            return False

        cond = (len(n['addresses']) > 0, # Node must be connectable
                graph.degree(nkey) > 0, # Must have unfiltered channels
                minchancount <= n['num_channels'] <= maxchancount,
                mincapacity <= n['capacity'] <= maxcapacity,
                n['capacity']/n['num_channels'] > minavgchan, # avg chan size
                )
        # Filters that require more computation are excluded from cond
        # in order to get a slight performance boost from short circuiting

        return (all(cond)
                and validatenodechancapacities(n['capacities'], minchanstiers)
                and determinereliability(nkey, graph) >= minreliability
                )

    return [n['pub_key'] for n in
            filter(lambda n: filtercandidatenodes(n, graph, filters),
                   graph.nodes.values())
           ]

def filterrelevantnodes(graph, graphfilters):
    minrelevantchan = graphfilters.getint('minrelevantchan')

    t = time.time()
    def filter_node(nk):
        n = graph.nodes[nk]

        cond = (
                # A node that hasn't updated in this time might be dead
                t - n['last_update'] < 4 * 30 *24*60*60 ,
                graph.degree(nk) > 0, # doesn't seem to fix the issue
            )

        return all(cond)

    def filter_edge(n1, n2):
        e = graph.edges[n1, n2]

        isours = mynodekey in [n1, n2]

        cond = (
                # A channel that hasn't updated in this time might be dead
                t - e['last_update'] <=  1.5 *24*60*60,

                # Remove economically irrelevant channels
                e['capacity'] >= minrelevantchan or isours,
                )

        return all(cond)



    gfilt = nx.subgraph_view(graph, filter_node=filter_node, filter_edge=filter_edge)

    mostrecentupdate = 0
    for e in gfilt.edges.values():
         if e['last_update'] > mostrecentupdate:
            mostrecentupdate = e['last_update']

    if t - mostrecentupdate > 8*60*60:
        raise RuntimeError('Graph is more than 8 hours old, results will be innaccurate')

    return gfilt

## Configuration ends here
def preparegraph(mynodekey, graphfilters):

    gfull = loadgraph.lnGraph.autoload()
    for newpeer in addchannels:
        gfull.add_edge(mynodekey, newpeer, capacity=newchannelsize, last_update=time.time())
    for newpeer in removechannels:
        gfull.remove_edge(mynodekey, newpeer)
    nx.freeze(gfull)

    print('Loaded graph. Number of nodes:', gfull.number_of_nodes(), 'Edges:', gfull.number_of_edges())
    g = filterrelevantnodes(gfull, graphfilters)

    print('Simplified graph. Number of nodes:', g.number_of_nodes(), 'Edges:', g.number_of_edges())
    if g.number_of_edges() == 0:
        raise RuntimeError('No recently updated channels were found, is describgraph.json recent?')

    return g

def calculatefarnessscore(peer2add, myfarness, graphcopy, mynodekey):

    # Modify the graph with a simulated channel
    graphcopy.add_edge(peer2add, mynodekey)

    mynewfarness = 1/fastcentrality.closeness(graphcopy, mynodekey)

    myfarnessdelta = mynewfarness - myfarness

    # Since this function is batched, and making a fresh copy is slow,
    # Make sure all changes are undone
    graphcopy.remove_edge(peer2add, mynodekey)

    # Want this data from the unmodified graph
    # Otherwise their score will be lowered if the channel
    # is too beneficial to them
    theirfarness = 1/fastcentrality.closeness(graphcopy, peer2add)

    # This is where the magic happens
    # Nodes that reduce our farness,
    # as well as nodes with a high farness,
    # are prioritized. But especially nodes that have both.
    farnessscore = np.cbrt(abs(myfarnessdelta)) + theirfarness/1000

    return farnessscore

def calculatefarnessscores(candidatekeys, graph, mynodekey):
    farnesscores = {}

    myfarness = 1/fastcentrality.closeness(graph, mynodekey)

    print('Running modified farness score calculations')
    t = time.time()
    with ProcessPoolExecutor() as executor:
        scoreresults = executor.map(calculatefarnessscore,
                                    candidatekeys,
                                    repeat(myfarness),
                                    repeat(nx.Graph(graph)),
                                    repeat(mynodekey),
                                    chunksize=128)

    for nkey, score in zip(candidatekeys, scoreresults):
        farnesscores[nkey] = score

    print(f'Completed modified farness score calculations in {time.time()-t:.1f}s')

    return farnesscores

def sortbyfarnesscore(candidatekeys, farnesscores):
    sortedkeys = sorted(candidatekeys, key=lambda k:-farnesscores[k])
    return sortedkeys

def get1mlcache(cached1ml={}):
    if cached1ml:
        pass

    elif os.path.isfile('_cache/1mlcache.json'):
        print('Found cache file for 1ML statistics')
        with open('_cache/1mlcache.json') as f:
            cached1ml.update(json.load(f))

    elif not os.path.exists('_cache'):
        os.mkdir('_cache')

    return cached1ml

def get1mlstats(node_key):
    cachetimeout =  3*24*60*60
    cached1ml = get1mlcache()

    if node_key in cached1ml.keys():
        node1ml = cached1ml[node_key]
        cutofftime = time.time() - cachetimeout
        if node1ml.get('_fetch_time', 0) > cutofftime:
            return node1ml
        # else, fetch a new copy

    r = requests.get(f"https://1ml.com/node/{node_key}/json")
    if r.status_code != 200:
        print(f'Bad response {r.status_code}: {r.body}')
        raise RuntimeError(f'Bad response {r.status_code}: {r.body}')
    cached1ml[node_key] = r.json()
    cached1ml[node_key]['_fetch_time'] = time.time()

    with open('_cache/1mlcache.json', 'w') as f:
        json.dump(cached1ml, f, indent=2)

    return cached1ml[node_key]

def filterbyavailability(candidatekeys, max1mlavailability):

    def checkavailablityscore(nodekey):
        nodestats1ml = get1mlstats(nodekey)
        try:
            availabilityrank = nodestats1ml['noderank']['availability']
        except KeyError:
            print('Failed to fetch availability for', nodekey)
            return False

        return availabilityrank < max1mlavailability

    results = filter(checkavailablityscore, candidatekeys)

    return results

def selectby1ml(sortedcandidates, max1mlavailability, finalcandidatecount):
    print('Checking 1ML availability statistics')
    t = time.time()
    availablecandidates = []
    for i, canditate in enumerate(
        filterbyavailability(sortedcandidates, max1mlavailability)):
            if i >= finalcandidatecount: break
            availablecandidates.append(canditate)

    print('1ML availability filter selected',
          len(availablecandidates),
          'candidates for new channels',
          f'in {time.time()-t:.1f}s')
    return availablecandidates

def calculatenewcentrality(peer2add, graphcopy, mynodekey):
    graphcopy.add_edge(peer2add, mynodekey)

    newbc = fastcentrality.betweenness(graphcopy, mynodekey)

    # Remove in case the same instance is reused due to batching
    graphcopy.remove_edge(peer2add, mynodekey)

    return newbc

def calculatemycentrality(graphcopy, mynodekey):
    return fastcentrality.betweenness(graphcopy, mynodekey)

def calculatecentralitydeltas(candidatekeys, graph, mynodekey):
    centralitydeltas = {}
    t = time.time()

    with ProcessPoolExecutor() as executor:
        print('Starting baseline centrality computation')
        mycentralityfuture = executor.submit(calculatemycentrality,
                                             nx.Graph(graph), mynodekey)

        print('Queuing computations for new centralities')

        newcentralities = executor.map(calculatenewcentrality,
                                        candidatekeys,
                                        repeat(nx.Graph(graph)),
                                        repeat(mynodekey),
                                        chunksize=4)


        print('Waiting for baseline centrality calculation to complete')
        myoldcentrality = mycentralityfuture.result()

        print('Our current centrality is approximately', int(myoldcentrality))

        print('Collecting centrality results, this may take a while')

        counter = 0
        njobs = len(candidatekeys)
        print(f'Progress: {counter}/{njobs} {counter/njobs:.1%}',
              f'Elapsed time {time.time()-t:.1f}s',
              end='\r')

        for nkey, newcentrality in zip(candidatekeys, newcentralities):

            centralitydelta = newcentrality - myoldcentrality
            centralitydeltas[nkey] = centralitydelta

            counter += 1
            print(f'Progress: {counter}/{njobs} {counter/njobs:.1%}',
                  f'Elapsed time {time.time()-t:.1f}s',
                  end='\r')


    print(f'Completed centrality difference calculations in {time.time()-t:.1f}s')
    return centralitydeltas, myoldcentrality

def printresults(centralitydeltas, mycurrentcentrality):
    cols = 'Δcentr','MFscor','Avail','Relbty','Alias','Pubkey'
    print(*cols)
    exportdict = {k:[] for k in cols}
    for nkey, cdelta in sorted(centralitydeltas.items(), key=lambda i:-i[1]):
        nodedata = g.nodes[nkey]
        alias = nodedata['alias']
        mfscore = farnesscores[nkey]
        arank = get1mlstats(nkey)['noderank']['availability']
        reliability = 1 - nodedata['disabledcount']['receiving']/g.degree(nkey)

        cdeltastr = f'{cdelta/mycurrentcentrality:6.1%}'
        relbtystr = f'{reliability:6.1%}'
        exportdict['Δcentr'].append(cdeltastr)
        exportdict['MFscor'].append(mfscore)
        exportdict['Avail'].append(arank)
        exportdict['Relbty'].append(relbtystr)
        exportdict['Alias'].append(alias)
        exportdict['Pubkey'].append(nkey)

        print(f'{cdelta/mycurrentcentrality:+6.1%} {mfscore:6.2f} {arank:5}',
                relbtystr, alias, nkey)

    return exportdict


if __name__ == '__main__':
    config = loadconfig()
    mynodekey = config['Node']['pub_key']
    filters = config['CandidateFilters']

    graphfilters = config['GraphFilters']
    g = preparegraph(mynodekey, graphfilters)

    print('Performing analysis for', g.nodes[mynodekey]['alias'])

    newchannelcandidates = selectinitialcandidates(g, filters)
    print('First filtering pass found', len(newchannelcandidates), 'candidates for new channels')

    farnesscores = calculatefarnessscores(newchannelcandidates, g, mynodekey)
    mfssortedcandidates = sortbyfarnesscore(newchannelcandidates, farnesscores)

    max1mlavailability = filters.getint('max1mlavailability')
    finalcandidatecount = filters.getint('finalcandidatecount')

    availablecandidates = selectby1ml(mfssortedcandidates, max1mlavailability,
                                      finalcandidatecount)

    if len(availablecandidates) == 0:
        print('No candidates found, is your graph stale?')
        print('If issue persists, delete describegraph.json and improvecentrality.conf')

    centralitydeltas, mycurrentcentrality = calculatecentralitydeltas(
                                          availablecandidates, g, mynodekey)
    exportdict = printresults(centralitydeltas, mycurrentcentrality)

    csvexportname = config['Other'].get('csvexportname')
    if csvexportname:
        df = pd.DataFrame(exportdict)
        df.set_index('Pubkey')
        df.to_csv(csvexportname)
