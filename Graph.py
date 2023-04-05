
from typing import Any, Dict, List, Optional


class Node:
    def __init__(self, name: str):
        self.name:str = name
        self.neighbors = {}

    def add_neighbor(self, neighbor, weight = 0):
        self.neighbors[neighbor] = weight

    def get_neighbors(self):
        return list(self.neighbors.keys())

    def get_weight(self, neighbor):
        return self.neighbors[neighbor]
    
    def number_of_edges(self):
        return len(self.neighbors)

    def __str__(self) -> str:
        return self.name + " <neighbors> : "  + str([neighbor.name for neighbor in self.neighbors.keys()])

class Graph:
    def __init__(self):
        self.nodes : Dict[str, Node]= {}

    def add_node(self, node: str) -> None:
        self.nodes[node] = Node(node)

    def get_cost(self, node1: str, node2: str) -> int|float:
        return self.nodes[node1].get_weight(self.nodes[node2])

    def add_edge(self, node1: str, node2: str, weight=0) -> None:
        if node1 not in self.nodes:
            self.add_node(node1)
        if node2 not in self.nodes:
            self.add_node(node2)
        self.nodes[node1].add_neighbor(self.nodes[node2], weight)
        self.nodes[node2].add_neighbor(self.nodes[node1], weight)

    def delete_node(self, node: str) -> None:
        if node in self.nodes:
            del self.nodes[node]
            for n in self.nodes:
                if node in self.nodes[n].get_neighbors():
                    del self.nodes[n].neighbors[node]

    def delete_edge(self, node1: str, node2: str) -> None:
        if node1 in self.nodes and node2 in self.nodes:
            del self.nodes[node1].neighbors[self.nodes[node2]]
            del self.nodes[node2].neighbors[self.nodes[node1]]

    def search(self, item) -> Node:
        return self.nodes[item]
    
    def __str__(self) -> str:
        lst = list()

        for node in self.nodes.values():
            lst.append(str(node))

        return str("\n-> ".join(lst))

