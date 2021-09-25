import requests
import numpy as np


class CandidateFilter:
    def __init__(self, graph, candidate_filters):
        # Conditions a node must meet to be considered for further connection analysis
        self.min_channels = candidate_filters.getint('minchancount')
        self.min_capacity = int(candidate_filters.getfloat('mincapacitybtc') * 1e8)
        self.max_channels = candidate_filters.getint('maxchancount')
        self.max_capacity = int(candidate_filters.getfloat('maxcapacitybtc') * 1e8)
        self.min_avg_chan = candidate_filters.getint('minavgchan')
        self.minmedchan = candidate_filters.getint('minmedchan')
        self.minavgchanageblks = candidate_filters.getint('minavgchanageblks')
        self.minreliability = candidate_filters.getfloat('minreliability')
        self.minchannelsstr = candidate_filters['minchannels']
        self.pub_key = candidate_filters['pub_key']
        self.minchanstiers = self.minchannelsstr.split()

        self.graph = graph

        assert self.minavgchanageblks is not None, 'Config is missing values, try recreating it'

        # Can't guarantee lnd available to get this
        self.currblockheight = requests.get(
            "https://mempool.space/api/blocks/tip/height"
        ).json()

        self.filtered_candidates = [n['pub_key'] for n in
                                    filter(lambda n: self.filtercandidatenodes(n),
                                           graph.nodes.values())]

    def validate_capacities(self, chancaps):
        """
        This function allows more granular selection of candidates than total or
        average capacity.
        """
        for tierfilter in self.minchanstiers:
            if 'k' in tierfilter:
                ksize, mincount = tierfilter.split('k')
                size = int(ksize) * 1e3
            elif 'M' in tierfilter:
                Msize, mincount = tierfilter.split('M')
                size = int(Msize) * 1e6
            else:
                raise RuntimeError('No recognized seperator in minchannel filter')

            if sum((c >= size for c in chancaps)) < int(mincount):
                return False

        return True

    def get_avg_change(self, candidatekey, currblockheight):
        ages = []
        for chankeypair in self.graph.edges(candidatekey):
            chan = self.graph.edges[chankeypair]
            chanblk = int(chan['channel_id']) >> 40
            ageblks = currblockheight - chanblk
            ages.append(ageblks)

        if len(ages) == 0:
            return -1

        return np.mean(ages)

    def get_reliability(self, candidatekey):
        # Record reliability-related data into the node record

        node = self.graph.nodes[candidatekey]

        # This should never happen, but there is evidence that it does
        if self.graph.degree(candidatekey) == 0:
            print(node)
            raise RuntimeError(f"{node['alias']} has no valid channels")

        chankeypairs = self.graph.edges(candidatekey)
        node['disabledcount'] = {'sending': 0, 'receiving': 0}

        for chankeypair in chankeypairs:
            chan = self.graph.edges[chankeypair]

            if None in [chan['node1_policy'], chan['node2_policy']]:
                continue  # TODO: Might want to add a penalty to this

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

        return 1 - node['disabledcount']['receiving'] / self.graph.degree(candidatekey)

    def filtercandidatenodes(self, n):
        # This is the inital filtering pass, it does not include 1ML or centrality

        # If already connected or is ourself, abort
        nkey = n['pub_key']
        if self.graph.has_edge(self.pub_key, nkey) or self.pub_key == nkey:
            return False

        cond = (len(n['addresses']) > 0,  # Node must be connectable
                self.graph.degree(nkey) > 0,  # Must have unfiltered channels
                self.min_channels <= n['num_channels'] <= self.max_channels,
                self.min_capacity <= n['capacity'] <= self.max_capacity,
                n['capacity'] / n['num_channels'] >= self.min_avg_chan,  # avg chan size
                np.median(n['capacities']) >= self.minmedchan,
                )

        # Filters that require more computation or are invalid for tiny
        # nodes are excluded from cond in order to get safety and a
        # slight performance boost from short circuiting

        return (all(cond)
                and self.validate_capacities(n['capacities'])
                and self.get_avg_change(nkey, self.currblockheight) >= self.minavgchanageblks
                and self.get_reliability(nkey) >= self.minreliability
                )
