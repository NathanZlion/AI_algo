import random
from undirectedGraph import Graph

class Graph_Generator:


    @staticmethod
    def generate(number_of_nodes: int, probability_of_edge: float):
        graph : Graph = Graph()
        total_possible_edges = (number_of_nodes) * (number_of_nodes-1) // 2
        needed_edge_amount = round(probability_of_edge*total_possible_edges)

        # create nodes
        for i in range(number_of_nodes):
            node_name = Graph_Generator._num_to_alpha(i)
            graph.add_node(node_name)

        stop = False
        for i in range(number_of_nodes - 1):
            if stop:
                break

            for j in range(i + 1, number_of_nodes):
                if needed_edge_amount == 0:
                    stop = True
                    break

                weight = random.uniform(0.1, 10.0)
                if Graph_Generator._add_edge(graph, i, j, weight):
                    needed_edge_amount -= 1

        coordinates = {}

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
        Changes a number to a string of characters, It's technically hashing. used \
        the concept of changing to base 26 and replacing each number to it's respective\
        character, A being 0  ,B => 1and goes like that. 
        """

        quotient = num // 26
        remainder = num % 26

        if quotient == 0:
            return chr(remainder+97)

        return Graph_Generator._num_to_alpha(quotient-1) + chr(remainder+97)


