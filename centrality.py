
from Searches import Search
from collections import defaultdict, deque
from romaniaCity import Romania
from queue import Queue, PriorityQueue
import heapq
from sys import maxsize
from graph import Graph, Node
from typing import List, Optional, Tuple, Dict
from math import radians, sqrt, sin, cos, atan2


class centrality:

    def degree_centrality(self, graph: Graph):
        """ this returns the centrality of all nodes in the input graph."""

        centrality = {}
        total_nodes = len(graph.nodes)

        # number of connections / n -1, n = total number of nodes
        for node in graph.nodes.values():
            try:
                centrality[node.name] = node.number_of_edges() /(total_nodes-1)
            except ZeroDivisionError:
                centrality[node.name] = 0

        return centrality

    def closeness_centrality(self, graph: Graph, coordinates):
        """for each node calculate the sum of shor"""

        search = Search()

        centrality = {}
        total_nodes = len(graph.nodes)

        for node in graph.nodes: centrality[node] = 0

        # for each node calculate the sum of shortest distance to all other nodes
        for start in graph.nodes:
            for end in graph.nodes:
                centrality[start] += search.get_path_cost(graph, search.a_star_search(graph, start, end, coordinates))
        
        for node in centrality:
            try:
                centrality[node] = (1/centrality[node]) * (total_nodes-1)
            except ZeroDivisionError:
                centrality[node] = 0

        return centrality

    def eigenvector_centrality():
        pass

    def katz_centrality():
        pass

    def pagerank_centrality():
        pass

    def betweenness_centrality():
        pass




center = centrality()

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

print(center.closeness_centrality(Romania().getCity(), romania_coordinates))
