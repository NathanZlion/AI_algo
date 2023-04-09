

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

        n = len(graph)
        paths = []
        centrality = {}

        for node in graph.get_nodes():
            centrality[node] = 0

        all_nodes = [node for node in graph.get_nodes()]

        # find every shortest path
        for i in range(n):
            for j in range(i, n):
                paths.append(Search().dijkstra_search(graph, all_nodes[i], all_nodes[j]))

        for path in paths:
            for index in range(1, len(path)-1):
                centrality[path[index]] += 1


        NORMALIZING_DIVIDER = (n) * (n - 1) / (2)

        for node in centrality.keys():
            centrality[node] /= NORMALIZING_DIVIDER

        return dict(centrality)


    def eigenvector_centrality(self, graph: Graph) -> Dict[str, float]:
        """Returns the `eigen vector centrality` of nodes in the graph."""
        
        # create adjacency matrix with weights as distances
        adj_matrix :List[list[int|float]] = [[0 for _ in range(len(graph))] for _ in range(len(graph))]
        node_names = list(graph.get_nodes().keys())
        node_indices = {name:index for index,name in enumerate(node_names)}

        edges = graph.get_edges()
        for edge in edges:
            source, target, weight = edge
            weight = 1/float(weight["weight"])
            row:int = node_indices[source]
            col:int = node_indices[target]
            adj_matrix[row][col] = weight

        # initialize eigenvector values
        ev_values = [1.0 for _ in range(graph.get_number_of_nodes())]
        convergence_threshold = 0.0001
        
        # power iteration method to calculate Eigenvector centrality
        while True:
            prev_ev_values = ev_values.copy()
            for i in range(graph.get_number_of_nodes()):
                ev_values[i] = sum(adj_matrix[i][j] * prev_ev_values[j] for j in range(graph.get_number_of_nodes()))
            norm_factor = max(ev_values)
            ev_values = [val / norm_factor for val in ev_values]
            if all(abs(ev_values[i] - prev_ev_values[i]) < convergence_threshold for i in range(graph.get_number_of_nodes())):
                break

        # reverse the centrality values
        max_centrality = max(ev_values)
        res = {node: max_centrality / value for node, value in zip(node_names, ev_values)}

        return res

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


