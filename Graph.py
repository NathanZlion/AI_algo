
from typing import Any, Dict, List, Optional, Tuple


class Node:
    def __init__(self, name: str):
        self.name:str = name
        self._neighbors: Dict['Node', int|float] = {}

    def add_neighbor(self, neighbor: 'Node', weight = 0):
        self._neighbors[neighbor] = weight

    def get_neighbors(self) -> Dict['Node', int|float]:
        """This returns a dictionary with key node and value int or float, which represents the weight to that node"""
        return self._neighbors

    def get_weight(self, neighbor: 'Node'):
        return self._neighbors[neighbor]

    def number_of_edges(self):
        return len(self._neighbors)

    def is_neighbor_of(self, node: 'Node'):
        return node in self._neighbors

    def __str__(self) -> str:
        return self.name + " <neighbors> : "  + str([neighbor.name for neighbor in self._neighbors.keys()])

class Graph:
    def __init__(self):
        self._nodes : Dict[str, Node]= {}

    def add_node(self, node: str) -> None:
        self._nodes[node] = Node(node)

    def get_cost(self, node1: str, node2: str) -> int|float:
        return self._nodes[node1].get_weight(self._nodes[node2])

    def add_edge(self, node1: str, node2: str, weight=0) -> None:
        if node1 not in self._nodes:
            self.add_node(node1)
        if node2 not in self._nodes:
            self.add_node(node2)
        self._nodes[node1].add_neighbor(self._nodes[node2], weight)
        self._nodes[node2].add_neighbor(self._nodes[node1], weight)

    def delete_node(self, node: str) -> None:
        if node in self._nodes:
            del self._nodes[node]
            for n in self._nodes:
                if node in self._nodes[n].get_neighbors():
                    del self._nodes[n].get_neighbors()[node]

    def delete_edge(self, node1: str, node2: str) -> None:
        if node1 in self._nodes and node2 in self._nodes:
            del self._nodes[node1].get_neighbors()[self._nodes[node2]]
            del self._nodes[node2].get_neighbors()[self._nodes[node1]]

    def search(self, item) -> Node:
        return self._nodes[item]
    
    def have_edge(self, node1: str, node2: str) -> bool:
        return self._nodes[node1].is_neighbor_of(self._nodes[node2])
    
    def get_nodes(self) -> Dict[str, Node]:
        return self._nodes

    def get_edges(self) -> List[Tuple[str, str, str|int|float]]:

        # intended output
        # [] ("Oradea", "Zerind", 71) ,  ...]

        weights : Dict[Tuple[str,str], int|float]= {}
        for node in self.get_nodes().values():
            for neighbor in node.get_neighbors():
                if not (node.name, neighbor.name) in weights and not (neighbor.name, node.name) in weights:
                    weights[(node.name, neighbor.name)] = node.get_weight(neighbor)
        
        res = [(node1, node2,weights[(node1, node2)]) for (node1, node2) in weights]

        return res

    def __str__(self) -> str:
        lst = list()

        for node in self._nodes.values():
            lst.append(str(node))

        return str("\n-> ".join(lst))

    def __len__(self) -> int:
        return len(self.get_nodes())

