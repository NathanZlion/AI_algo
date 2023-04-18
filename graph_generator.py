import random
from typing import Dict, Tuple
from undirected_graph import Graph

class Graph_Generator:
    """A module to generate a random graph.
    ---
    Methods:
        - `generate(number_of_nodes: int, probability_of_edge: float = 0.5)` : generates a graph randomly and returns it and it's coordinate.
                - Takes as an argument the number of nodes needed and the probability of edge needed (50% default value).
                - It generates random nodes of the given count.
                - Then assigns random coordinates for it.
                - returns a tuple of the graph : `Graph`, and coordinates : Dict[str, Tuple[float, float]]

    """

    @staticmethod
    def generate(number_of_nodes: int, probability_of_edge: float = 0.5) -> Tuple[Graph, Dict[str, Tuple[float, float]]]:
        """
        Generates graph with `number of nodes` and `probability of edge`as given in parameter.
        Also assigned a random (x,y) coordinate value for the nodes for later use in heuristics.

        """
        graph = Graph()
        total_possible_edges = (number_of_nodes) * (number_of_nodes-1) // 2
        needed_edge_amount = round(probability_of_edge*total_possible_edges)

        # create nodes
        for i in range(number_of_nodes):
            node_name = Graph_Generator._num_to_alpha(i)
            graph.add_node(node_name)

        stop = False
        for i in range(number_of_nodes - 1):
            for j in range(i + 1, number_of_nodes):
                if needed_edge_amount == 0:
                    stop = True
                    break

                weight = random.uniform(0.1, 10.0)
                if Graph_Generator._add_edge(graph, i, j, weight):
                    needed_edge_amount -= 1

            if stop:
                break

        coordinates : Dict[str, Tuple[float, float]] = {}

        for node in graph.get_nodes():
            x = random.uniform(0, 1)*10
            y = random.uniform(0, 1)*10
            coordinates[node] = (x, y)


        return (graph, coordinates)


    @staticmethod
    def _add_edge(graph: Graph, i: int, j: int, weight: float) -> bool:
        first_node = Graph_Generator._num_to_alpha(i)
        second_node = Graph_Generator._num_to_alpha(j)
        if graph.has_edge(first_node, second_node) or first_node==second_node:
            return False

        graph.add_edge(first_node, second_node, weight=weight)
        return True


    @staticmethod
    def _num_to_alpha(num) -> str:
        """
        Changes a number to a unique string of characters. It's technically hashing. used \
        the concept of changing to base 26 and replacing each number to it's respective \
        character, `A => 0`, `B => 1` and goes like that. 
        """

        quotient = num // 26
        remainder = num % 26

        if quotient == 0:
            return chr(remainder+97)

        return Graph_Generator._num_to_alpha(quotient-1) + chr(remainder+97)


