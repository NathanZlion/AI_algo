
from collections import defaultdict
from decimal import Decimal
from Searches import Search
from romaniaCity import Romania
from queue import Queue, PriorityQueue
import heapq
from sys import maxsize
from graph import Graph, Node
from typing import List, Optional, Tuple, Dict
import networkx as nx
import numpy as np


class centrality:

    def degree_centrality(self, graph: Graph):
        """ this returns the centrality of all nodes in the input graph."""

        centrality = {}
        total_nodes = len(graph.get_nodes())

        # number of connections / n -1, n = total number of nodes
        for node in graph.get_nodes().values():
            try:
                centrality[node.name] = node.number_of_edges() /(total_nodes-1)
            except ZeroDivisionError:
                centrality[node.name] = 0

        return centrality

    def closeness_centrality(self, graph: Graph, coordinates):
        """for each node calculates the sum of shortest paths and averages them."""

        search = Search()

        centrality = {}
        total_nodes = len(graph.get_nodes())

        for node in graph.get_nodes(): centrality[node] = 0

        # for each node calculate the sum of shortest distance to all other nodes
        for start in graph.get_nodes():
            for end in graph.get_nodes():
                centrality[start] += search.get_path_cost(graph, search.a_star_search(graph, start, end, coordinates))
        
        for node in centrality:
            if centrality[node] != 0:
                centrality[node] = (1/centrality[node]) * (total_nodes-1)
            else:
                centrality[node] = maxsize

        return centrality

    def eigenvector_centrality(self, graph: Graph):

        """returns the eigen vector centrality of nodes in the graph."""
        # load the Romania graph
        edges = graph.get_edges()

        # create a new graph with weights as distances
        dist_graph = nx.Graph()
        dist_graph.add_weighted_edges_from(edges)

        # calculate Eigenvector centrality
        eigenvector_centrality = nx.eigenvector_centrality_numpy(dist_graph)

        # reverse the centrality values
        max_centrality = max(eigenvector_centrality.values())
        eigenvector_centrality = {node: max_centrality - value for node, value in eigenvector_centrality.items()}

        return eigenvector_centrality

    def betweenness_centrality(self, graph: Graph, romania_coordinates):

        paths = []
        centrality = {}

        for node in graph.get_nodes():
            centrality[node] = 0

        search = Search()
        
        all_nodes = [node for node in graph.get_nodes()]

        # find every shortest path
        for i in range(len(graph)):
            for j in range(i, len(graph)):
                # paths.append(search.a_star_search(graph, all_nodes[i], all_nodes[j], romania_coordinates))
                paths.append(search.dijkstra(graph, all_nodes[i], all_nodes[j]))

        for path in paths:
            for index in range(1, len(path)-1):
                centrality[path[index]] += 1

        n = len(graph)

        NORMALIZING_DIVIDER = (n - 1) * (n - 2) / (2)

        for node in centrality.keys():
            centrality[node] /= NORMALIZING_DIVIDER

        return dict(centrality)

    # def katz_centrality():
    #     pass

    # def pagerank_centrality():
    #     pass



# central = centrality().eigenvector_centrality(Romania().get_city())


romania_coordinates : dict[str, Tuple[float, float]] = {
    "Arad": (46.18656, 21.31227),
    "Bucharest": (44.42676, 26.10254),
    "Craiova": (44.31813, 23.80450),
    "Drobeta": (44.62524, 22.65608),
    "Eforie": (44.06562, 28.63361),
    "Fagaras": (45.84164, 24.97264),
    "Giurgiu": (43.90371, 25.96993),
    "Hirsova": (44.68935, 27.94566),
    "Iasi": (47.15845, 27.60144),
    "Lugoj": (45.69099, 21.90346),
    "Mehadia": (44.90411, 22.36452),
    "Neamt": (46.97587, 26.38188),
    "Oradea": (47.05788, 21.94140),
    "Pitesti": (44.85648, 24.86918),
    "Rimnicu Vilcea": (45.10000, 24.36667),
    "Sibiu": (45.79833, 24.12558),
    "Timisoara": (45.75972, 21.22361),
    "Urziceni": (44.71667, 26.63333),
    "Vaslui": (46.64069, 27.72765),
    "Zerind": (46.62251, 21.51742)
}

central = centrality().betweenness_centrality(Romania().get_city(), romania_coordinates)
print(central["Oradea"])




















# print(center.closeness_centrality(Romania().get_city(), romania_coordinates))

# first = center.eigenvector_centrality(Romania().get_city())


# print(first)

# romania_edges = [
#     ("Oradea", "Zerind", {"weight": 71}),
#     ("Oradea", "Sibiu", {"weight": 151}),
#     ("Zerind", "Arad", {"weight": 75}),
#     ("Arad", "Sibiu", {"weight": 140}),
#     ("Arad", "Timisoara", {"weight": 118}),
#     ("Timisoara", "Lugoj", {"weight": 111}),
#     ("Lugoj", "Mehadia", {"weight": 70}),
#     ("Mehadia", "Drobeta", {"weight": 75}),
#     ("Drobeta", "Craiova", {"weight": 120}),
#     ("Craiova", "Pitesti", {"weight": 138}),
#     ("Craiova", "Rimnicu Vilcea", {"weight": 146}),
#     ("Rimnicu Vilcea", "Sibiu", {"weight": 80}),
#     ("Sibiu", "Fagaras", {"weight": 99}),
#     ("Fagaras", "Bucharest", {"weight": 211}),
#     ("Pitesti", "Bucharest", {"weight": 101}),
#     ("Rimnicu Vilcea", "Pitesti", {"weight": 97}),
#     ("Urziceni", "Bucharest", {"weight": 85}),
#     ("Urziceni", "Vaslui", {"weight": 142}),
#     ("Vaslui", "Iasi", {"weight": 92}),
#     ("Iasi", "Neamt", {"weight": 87}),
#     ("Urziceni", "Hirsova", {"weight": 98}),
#     ("Hirsova", "Eforie", {"weight": 86})
# ]
        
