
from random import sample
from typing import List
from Searches import Search
from undirectedGraph import Graph
from time import perf_counter as prf_ctr


class Experiment:


    @staticmethod
    def run(graph: Graph, coordinates, number_of_experiments: int = 10):
        """returns runtime and solution length."""

        search_algorithms = ["A_star", "breadth_first_search", "bidirectional_search", "dijkstra_search", \
                            "uniform_cost_search", "iterative_deepening_search", "depth_first_search", \
                            "greedy_search"]

        search_algorithm_name_to_function_map = {
            "A_star" : Search.a_star_search,
            "breadth_first_search" : Search.bfs,
            "bidirectional_search" : Search.bidirectional_search,
            "dijkstra_search" : Search.dijkstra_search,
            "uniform_cost_search" : Search.ucs,
            "iterative_deepening_search" : Search.ids,
            "depth_first_search" : Search.dfs_recursive,
            "greedy_search" : Search.greedy_search
        }

        search_runtime = [0.0 for _ in range(8)]
        search_solution_length = [0.0 for _ in range(8)]
        randomlist = Experiment.take_sample(list(graph.get_nodes().keys()))

        for _ in range(number_of_experiments):
            for start in randomlist:
                for goal in randomlist:
                    if start == goal: continue

                    for search_index in range(8):
                        search_algorithm = search_algorithm_name_to_function_map[search_algorithms[search_index]]

                        if search_index == 0:
                            start_timer = prf_ctr()
                            solution = search_algorithm(graph, start, goal, coordinates)
                            end_timer = prf_ctr()
                            search_runtime[0] += (end_timer - start_timer)
                            search_solution_length[0] += len(solution)
                        else:
                            start_timer = prf_ctr()
                            solution = search_algorithm(graph, start, goal)
                            end_timer = prf_ctr()
                            search_runtime[search_index] += (end_timer - start_timer)
                            search_solution_length[search_index] += len(solution)

        number_of_paths = (graph.get_number_of_nodes()*(graph.get_number_of_nodes()-1))/2

        for index in range(8):
            search_runtime[index] /= number_of_paths + number_of_experiments * 1000000
            search_solution_length[index] /= number_of_paths + number_of_experiments

        return (search_runtime, search_solution_length)


    @staticmethod
    def take_sample(nodes: List[str], number_of_samples = 10) -> List[str]:
        randomlist = sample(nodes, number_of_samples)

        return randomlist
