

from Searches import Search
from sys import maxsize
from romaniaCity import Romania
from undirectedGraph import Graph, Node
from typing import List, Optional, Tuple, Dict
import networkx as nx


class Centrality:
    def degree_centrality(self, graph: Graph):
        """ This returns the degree centrality of all nodes in the input graph."""

        centrality : Dict[str, float]= {}
        total_nodes = graph.get_number_of_nodes()

        for node in graph.get_nodes().values():
            centrality[node.name] = node.number_of_edges() / (total_nodes-1) if (total_nodes-1) > 0 else 0

        return centrality

    def closeness_centrality(self, graph: Graph):
        """
            For each node calculates the sum of shortest paths and averages them. \
            That gives us the closeness centrality. a higher centrality implying \
            more importance and a low one implying less importance.
        """

        search = Search()

        centrality = {}
        total_nodes = graph.get_number_of_nodes()

        for node in graph.get_nodes():
            centrality[node] = 0

        # for each node calculate the sum of shortest distance to all other nodes
        for source in graph.get_nodes():
            for destination in graph.get_nodes():
                centrality[source] += search.get_path_cost(graph, search.dijkstra_search(graph, source, destination))
        
        for node in centrality:
            centrality[node] = maxsize if centrality[node] == 0 else (total_nodes-1) * (1/centrality[node])

        return centrality

    def betweenness_centrality(self, graph: Graph):
        """ 
            This returns the `betweeness centrality` of all nodes in the input graph, by calculating\
            how many shortest paths pass through the current node. The more nodes pass through it \
            the more important (central) it will be.

        """

        paths = []
        centrality = {}

        for node in graph.get_nodes():
            centrality[node] = 0

        search = Search()

        all_nodes = [node for node in graph.get_nodes()]

        # find every shortest path
        for i in range(len(graph)):
            for j in range(i, len(graph)):
                paths.append(search.dijkstra_search(graph, all_nodes[i], all_nodes[j]))

        for path in paths:
            for index in range(1, len(path)-1):
                centrality[path[index]] += 1

        n = len(graph)

        NORMALIZING_DIVIDER = (n) * (n - 1) / (2)

        for node in centrality.keys():
            centrality[node] /= NORMALIZING_DIVIDER

        return dict(centrality)
    def eigenvector_centrality(self, graph: Graph):
        """Returns the `eigen vector centrality` of nodes in the graph."""

        # load the Romania graph
        # edges = graph.get_inverted_edges()
        edges = graph.get_edges()

        # create a new graph with weights as distances
        dist_graph = nx.Graph()
        for edge in edges:
            source, target, weight = edge
            dist_graph.add_edge(source, target, weight=weight)

        # calculate Eigenvector centrality
        eigenvector_centrality = nx.eigenvector_centrality_numpy(dist_graph)

        # reverse the centrality values
        max_centrality = max(eigenvector_centrality.values())
        eigenvector_centrality = {node: max_centrality - value for node, value in eigenvector_centrality.items()}

        return eigenvector_centrality


    def katz_centrality(self, graph: Graph, alpha:float|int = 0.1, max_iter: int = 100):
        # node_to_index_map is the map to the index in row and col of adj_matrix that the node has

        current_graph = nx.Graph()
        graph_edges = graph.get_inverted_edges()
        current_graph.add_edges_from(graph_edges)
        katz_centrality = nx.katz_centrality(current_graph, alpha=alpha, max_iter=max_iter)

        return katz_centrality

    def pagerank_centrality(self, graph: Graph, alpha= 0.85):
        current_graph = nx.Graph()
        graph_edges = graph.get_inverted_edges()
        current_graph.add_edges_from(graph_edges)

        pr = nx.pagerank(current_graph, alpha=alpha)
        
        return pr

