import os
import argparse
import json
import time
import requests
import configparser


def load_config(config_file='improvecentrality.conf'):
    if os.path.isfile(config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        return config

    print('Config not found, will create', config_file)
    config = configparser.ConfigParser()
    config['Node'] = {'pub_key': 'yournodepubkeyhere'}
    config['GraphFilters'] = {
        # Ignore channels smaller than this during analysis,
        # unless they connect to us. Higher values improve
        # script performance and odds of good routes.
        # However lower values give numbers closer to reality
        'minrelevantchan': 500_000,
    }
    config['CandidateFilters'] = {
        'minchancount': 8,
        'maxchancount': 10000,
        'mincapacitybtc': 0.3,
        'maxcapacitybtc': 1000,
        'minavgchan': 1_000_000,
        'minmedchan': 1_000_000,
        'minavgchanageblks': 4_000,
        # Default 4+ >1.5M channels, 2+ >3M channels
        'minchannels': '1500k4 3M2',
        'minreliability': 0.97,
        # Node must be ranked better than this for availability on 1ml
        'max1mlavailability': 2000,
        # Limit the number of nodes passed to the final (slow!) centrality computation
        'finalcandidatecount': 11,
    }
    config['Other'] = {
        # Export results to this CSV file. Set to None or '' to disable
        'csvexportname': 'newchannels.csv',
    }

    with open(config_file, 'w') as cf:
        config.write(cf)

    print('Please complete the config and rerun')
    exit()


def make_parser():
    parser = argparse.ArgumentParser(description='Suggests peers based on centrality impact')
    parser.add_argument('--conffile', type=str, default='improvecentrality.conf',
                        help='Specify an alternate config location')
    parser.add_argument('--validate', action="store_true",
                        help='Runs a much longer analysis and reports on the heuristic')
    return parser


def get_1ml_cache(cached1ml=None):
    if cached1ml is None:
        cached1ml = dict()
    if cached1ml:
        pass

    elif os.path.isfile('_cache/1mlcache.json'):
        # print('Found cache file for 1ML statistics')
        with open('_cache/1mlcache.json') as f:
            cached1ml.update(json.load(f))

    elif not os.path.exists('_cache'):
        os.mkdir('_cache')

    return cached1ml


def get_1ml_stats(node_key):
    cachetimeout = 3 * 24 * 60 * 60
    cached1ml = get_1ml_cache()

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


def filter_by_availability(candidatekeys, max1mlavailability):
    def checkavailablityscore(nodekey):
        nodestats1ml = get_1ml_stats(nodekey)
        try:
            availabilityrank = nodestats1ml['noderank']['availability']
        except KeyError:
            print('Failed to fetch availability for', nodekey)
            return False

        return availabilityrank < max1mlavailability

    results = filter(checkavailablityscore, candidatekeys)

    return results


def select_by_1ml(sortedcandidates, max1mlavailability, finalcandidatecount):
    print('Checking 1ML availability statistics')
    t = time.time()
    availablecandidates = []
    for i, canditate in enumerate(
            filter_by_availability(sortedcandidates, max1mlavailability)):
        if i >= finalcandidatecount: break
        availablecandidates.append(canditate)

    print('1ML availability filter selected',
          len(availablecandidates),
          'candidates for new channels',
          f'in {time.time() - t:.1f}s')
    return availablecandidates
