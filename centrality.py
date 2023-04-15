

import numpy as np
from Searches import Search
from sys import maxsize
from undirectedGraph import Graph
from typing import List, Dict, Union
import networkx as nx


class Centrality:

    @staticmethod
    def degree_centrality(graph: Graph):
        """ This returns the degree centrality of all nodes in the input graph."""

        centrality : Dict[str, float]= {}
        total_nodes = graph.get_number_of_nodes()

        for node in graph.get_nodes().values():
            centrality[node.name] = node.number_of_edges() / (total_nodes-1) if (total_nodes-1) > 0 else 0

        return centrality


    @staticmethod
    def closeness_centrality(graph: Graph):
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


    @staticmethod
    def betweenness_centrality(graph: Graph):
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


    @staticmethod
    def eigenvector_centrality(graph: Graph, max_iteration = 1000) -> Dict[str, float]:
        """Returns the `eigen vector centrality` of nodes in the graph."""

        # create adjacency matrix with weights as distances
        adj_matrix: List[List[Union[int, float]]] = [[0 for _ in range(len(graph))] for _ in range(len(graph))]
        node_names = list(graph.get_nodes().keys())
        node_indices = {name:index for index,name in enumerate(node_names)}

        edges = graph.get_edges()
        for edge in edges:
            source, target, weight = edge
            # used the inverse of weight as the weight signifies length of road connecting 
            # cities, so the further the less central. (inverse relationship)
            weight = 1/float(weight["weight"])
            row = node_indices[source]
            col = node_indices[target]
            adj_matrix[row][col] = weight
            adj_matrix[col][row] = weight

        # initialize eigenvector values
        ev_values = [1.0 for _ in range(graph.get_number_of_nodes())]
        convergence_threshold = 0.0001
        
        # power iteration method to calculate Eigenvector centrality with a maximum number of iterations
        for _ in range(max_iteration):
            prev_ev_values = ev_values.copy()
            for i in range(graph.get_number_of_nodes()):
                ev_values[i] = sum(adj_matrix[i][j] * prev_ev_values[j] for j in range(graph.get_number_of_nodes()))
            norm_factor = max(ev_values)
            ev_values = [val / norm_factor if norm_factor != 0 else 0 for val in ev_values]
            if all(abs(ev_values[i] - prev_ev_values[i]) < convergence_threshold for i in range(graph.get_number_of_nodes())):
                break

        # calculate the norm factor after convergence
        norm_factor = sum(ev_values)

        res = {node: norm_factor * value if value != 0 else 0 for node, value in zip(node_names, ev_values)}

        return res


    @staticmethod
    def katz_centrality(graph: Graph, alpha: float = 39.9, beta:float=1, max_iteration: int = 1000):
        """Returns the `katz centrality` of nodes in the graph."""
        
        adj_matrix: List[List[Union[int, float]]] = [[0 for _ in range(len(graph))] for _ in range(len(graph))]
        node_names = list(graph.get_nodes().keys())
        node_indices = {name:index for index,name in enumerate(node_names)}

        edges = graph.get_edges()
        for edge in edges:
            source, target, weight = edge
            # used the inverse of weight as the weight signifies length of road connecting 
            # cities, so the further the less central. (inverse relationship)
            weight = 1/float(weight["weight"])
            row = node_indices[source]
            col = node_indices[target]
            adj_matrix[row][col] = weight
            adj_matrix[col][row] = weight
            
        # calculate the maximum eigenvalue of the adjacency matrix
        max_eigenvalue = max(abs(eigenvalue) for eigenvalue in np.linalg.eigvals(adj_matrix))

        # initialize kv values
        kv_values = [0.0 for _ in range(graph.get_number_of_nodes())]
        convergence_threshold = 0.001 # defining convergence threshold as a small number
        
        # power iteration method to calculate Katz centrality
        for _ in range(max_iteration):
            prev_kv_values = kv_values.copy()
            for i in range(graph.get_number_of_nodes()):
                kv_values[i] = beta + alpha * sum(adj_matrix[i][j] * prev_kv_values[j] for j in range(graph.get_number_of_nodes()))
            # normalize kv values
            norm_kv_factor = max(kv_values)
            kv_values = [val / norm_kv_factor if norm_kv_factor != 0 else 0 for val in kv_values]
            if all(abs(kv_values[i] - prev_kv_values[i]) < convergence_threshold for i in range(graph.get_number_of_nodes())):
                break
        
        res = {node: value for node, value in zip(node_names, kv_values)}

        return res

    @staticmethod
    def pagerank_centrality(graph: Graph, alpha=0.85):
        """Returns the `pagerank centrality` of nodes in the graph."""
        assert  len(graph) > 0, "The graph has no nodes, cannot calculate centrality."

        # create adjacency matrix with weights as distances
        adj_matrix :List[list[int|float]] = [[0 for _ in range(len(graph))] for _ in range(len(graph))]
        node_names = list(graph.get_nodes().keys())
        node_indices = {name:index for index,name in enumerate(node_names)}
        convergence_threshold = 1e-5 # defining convergence threshold as a small number
        max_iter = 100 # defining maximum number of iterations as 100

        edges = graph.get_edges()
        outlinks_count = [0] * len(graph)
        for edge in edges:
            source, target, weight = edge
            weight = 1/float(weight["weight"])
            row:int = node_indices[source]
            col:int = node_indices[target]
            adj_matrix[row][col] = weight
            outlinks_count[row] += 1
        
        # initialize pagerank values
        pagerank_values = [1/len(graph)] * len(graph)

        # power iteration algorithm to calculate Pagerank
        i = 0
        while i < max_iter:
            prev_pagerank_values = pagerank_values.copy()
            for j in range(len(graph)):
                incoming_pr = sum(adj_matrix[i][j]/outlinks_count[j]*prev_pagerank_values[j] if outlinks_count[j]*prev_pagerank_values[j] != 0 else 0 for i in range(len(graph)))
                pagerank_values[j] = (1 - alpha) * 1/len(graph) + alpha * incoming_pr
            if all(abs(pagerank_values[i] - prev_pagerank_values[i]) < convergence_threshold for i in range(len(graph))):
                break  
            i += 1

        res = {node: value for node, value in zip(node_names, pagerank_values)}

        return res
