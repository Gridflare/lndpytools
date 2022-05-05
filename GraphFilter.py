import networkx as nx
import time


class GraphFilter:
    # Assumed size of the new channels, should not currently have an effect
    channel_size = 2e6

    def __init__(self, graph, mynodekey, graph_filters, channels_to_add=None, channels_to_remove=None):
        self.graph = graph
        self.graph_filters = graph_filters
        self.pub_key = mynodekey
        # Configuration ends here
        # self.full_g = self.autoload(expirehours=False)
        if channels_to_add:
            for peer in channels_to_add:
                self.graph.add_edge(mynodekey, peer, capacity=self.channel_size, last_update=time.time())
        if channels_to_remove:
            for peer in channels_to_remove:
                self.graph.remove_edge(mynodekey, peer)
        nx.freeze(self.graph)

        print('Loaded graph. Number of nodes:', self.graph.number_of_nodes(), 'Edges:', self.graph.number_of_edges())
        self.filtered_g = self.filter_relevant_nodes()

        print('Simplified graph. Number of nodes:', self.filtered_g.number_of_nodes(), 'Edges:',
              self.filtered_g.number_of_edges())
        if self.filtered_g.number_of_edges() == 0:
            raise RuntimeError('No recently updated channels were found, is describgraph.json recent?')

    def filter_relevant_nodes(self):
        t = time.time()
        gfilt = nx.subgraph_view(self.graph, filter_node=self.filter_node, filter_edge=self.filter_edge)

        most_recent_update = 0
        for edge in self.graph.edges.values():
            if most_recent_update < edge['last_update'] < t:
                most_recent_update = edge['last_update']

        print(f'Latest update in graph: {time.ctime(most_recent_update)} ({most_recent_update})')
        if t - most_recent_update > 6 * 60 * 60:
            raise RuntimeError('Graph is more than 6 hours old, results will be innaccurate')

        return gfilt

    def filter_node(self, nk):
        n = self.graph.nodes[nk]
        t = time.time()

        cond = (
            # A node that hasn't updated in this time might be dead
            t - n['last_update'] < 4 * 30 * 24 * 60 * 60,
            self.graph.degree(nk) > 0,  # doesn't seem to fix the issue
        )

        return all(cond)

    def filter_edge(self, n1, n2):
        minimum_capacity = self.graph_filters.getint('minrelevantchan')
        t = time.time()

        edge = self.graph.edges[n1, n2]
        is_adj = self.pub_key in [n1, n2]
        cond = (
            # A channel that hasn't updated in this time might be dead
            t - edge['last_update'] <= 1.5 * 24 * 60 * 60,

            # Remove economically irrelevant channels
            edge['capacity'] >= minimum_capacity or is_adj,
        )
        return all(cond)
