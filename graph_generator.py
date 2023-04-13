import random
from undirectedGraph import Graph

class Graph_Generator:
    
    @staticmethod
    def generate(number_of_nodes: int, probability_of_edge: float) -> Graph:
        resulting_graph : Graph = Graph()
        total_possible_edges = (number_of_nodes) * (number_of_nodes-1) // 2
        needed_edge_amount = round(probability_of_edge*total_possible_edges)

        # create nodes
        for i in range(number_of_nodes):
            node_name = Graph_Generator.num_to_alpha(i)
            resulting_graph.add_node(node_name)

        # add needed amount of edges
        all_nodes = list(resulting_graph.get_nodes().keys())

        # all_nodes = set(resulting_graph.get_nodes().keys())

        stop = False
        for i in range(number_of_nodes - 1):
            if stop:
                break

            for j in range(i + 1, number_of_nodes):
                if needed_edge_amount == 0:
                    stop = True
                    break

                if Graph_Generator._add_edge(resulting_graph, i, j):
                    needed_edge_amount -= 1

        return resulting_graph

    @staticmethod
    def _add_edge(graph: Graph, i: int, j: int) -> bool:
        first_node = Graph_Generator.num_to_alpha(i)
        second_node = Graph_Generator.num_to_alpha(j)
        if graph.has_edge(first_node, second_node) or first_node==second_node:
            return False

        graph.add_edge(first_node, second_node)
        return True

    @staticmethod
    def num_to_alpha(num) -> str:
        alphabet_map = lambda num: chr(num+97)

        quotient = num // 26
        remainder = num % 26
        if quotient == 0:
            return alphabet_map(remainder)

        return Graph_Generator.num_to_alpha(quotient-1) + alphabet_map(remainder)
    


print(Graph_Generator.generate(5, 0.5))