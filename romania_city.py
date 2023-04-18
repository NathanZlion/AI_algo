
import re
from typing import Dict, Tuple
from undirected_graph import Graph


class Romania:
    def __init__(self):
        self.romania = Graph()

        # add cities and connections(edges) between cities
        self.romania.add_edge("Oradea", "Zerind", 71)
        self.romania.add_edge("Oradea", "Sibiu", 151)
        self.romania.add_edge("Zerind", "Arad", 75)
        self.romania.add_edge("Arad", "Timisoara", 118)
        self.romania.add_edge("Timisoara", "Lugoj", 111)
        self.romania.add_edge("Lugoj", "Mehadia", 70)
        self.romania.add_edge("Mehadia", "Drobeta", 75)
        self.romania.add_edge("Drobeta", "Craiova", 120)
        self.romania.add_edge("Craiova", "Pitesti", 138)
        self.romania.add_edge("Arad", "Sibiu", 140)
        self.romania.add_edge("Sibiu", "Fagaras", 99)
        self.romania.add_edge("Sibiu", "Rimnicu Vilcea", 80)
        self.romania.add_edge("Rimnicu Vilcea", "Craiova", 146)
        self.romania.add_edge("Rimnicu Vilcea", "Pitesti", 97)
        self.romania.add_edge("Fagaras", "Bucharest", 211)
        self.romania.add_edge("Pitesti", "Bucharest", 101)
        self.romania.add_edge("Bucharest", "Giurgiu", 90)
        self.romania.add_edge("Bucharest", "Urziceni", 85)
        self.romania.add_edge("Urziceni", "Vaslui", 142)
        self.romania.add_edge("Urziceni", "Hirsova", 98)
        self.romania.add_edge("Hirsova", "Eforie", 86)
        self.romania.add_edge("Vaslui", "Iasi", 92)
        self.romania.add_edge("Iasi", "Neamt", 87)

    def get_city(self):
        return self.romania

    def get_coordinates(self) -> Dict[str, Tuple[float, float]] :
        romania_coordinates : dict[str, Tuple[float, float]] = {}

        # reading the coordinates from coordinates.txt file `
        with open("coordinates.txt", "r") as ef:
                for edges in ef:
                    edges_content = re.split('[:\n]', edges)
                    edges_content.pop()
                    edges_content = edges_content[0].split('    ')
                    romania_coordinates[edges_content[0]] = (float(edges_content[1]), float(edges_content[2]))
        
        return romania_coordinates



