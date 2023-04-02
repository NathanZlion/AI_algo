
from typing import Any, Dict, List


class Node:
    def __init__(self, name):
        self.name = name
        self.neighbors = {}

    def add_neighbor(self, neighbor, weight = 0):
        self.neighbors[neighbor] = weight

    def get_neighbors(self):
        return self.neighbors.keys()

    def get_weight(self, neighbor):
        return self.neighbors[neighbor]

    def __str__(self) -> str:
        # return self.name
        return self.name + " >>> "  + str([neighbor.name for neighbor in self.neighbors.keys()])

class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        self.nodes[node.name] = node

    def add_edge(self, node1, node2, weight=0):
        if node1 not in self.nodes:
            self.add_node(Node(node1))
        if node2 not in self.nodes:
            self.add_node(Node(node2))
        self.nodes[node1].add_neighbor(self.nodes[node2], weight)
        self.nodes[node2].add_neighbor(self.nodes[node1], weight)

    def delete_node(self, node):
        if node in self.nodes:
            del self.nodes[node]
            for n in self.nodes:
                if node in self.nodes[n].get_neighbors():
                    del self.nodes[n].neighbors[node]

    def delete_edge(self, node1, node2):
        if node1 in self.nodes and node2 in self.nodes:
            del self.nodes[node1].neighbors[self.nodes[node2]]
            del self.nodes[node2].neighbors[self.nodes[node1]]

    def search(self, item):
        try:
            return self.nodes[item]
        except KeyError:
            pass

    def __str__(self) -> str:
        lst = list()
        for nodename, node in self.nodes.items():
            lst.append(str(node))

        return str("\n".join(lst))



