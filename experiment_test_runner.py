
from random import sample
from typing import Callable, Dict, List
from romania_city import Romania
from searches import Search
from undirected_graph import Graph
from time import perf_counter


class Experiment:
    search_algorithm_names = ["a_star", "breadth_first_search", "greedy_search", "uniform_cost_search", "depth_first_search"]

    search_algorithm_name_to_func = {
        "a_star": Search.a_star_search,
        "breadth_first_search": Search.bfs,
        "greedy_search": Search.greedy_search,
        "uniform_cost_search": Search.ucs,
        "depth_first_search": Search.dfs_recursive,
    }

    @staticmethod
    def run(graph: Graph, coordinates, number_of_trial: int = 10):
        """returns runtime and solution length."""

        search_algorithm_names = Experiment.search_algorithm_names
        search_algorithm_name_to_func = Experiment.search_algorithm_name_to_func

        number_of_nodes = graph.get_number_of_nodes()
        search_runtime = [0.0 for _ in range(len(search_algorithm_names))]
        search_solution_length = [0.0 for _ in range(len(search_algorithm_names))]
        search_solution_cost = [0.0 for _ in range(len(search_algorithm_names))]
        randomlist = Experiment.__take_sample_nodes_from_graph(graph)
        heuristics = Search.calculate_heuristics(graph, coordinates)

        for _ in range(number_of_trial):
            for start in randomlist:
                for goal in randomlist:
                    if start == goal:
                        continue

                    # run all 8 searching algorithms and record their result.
                    for index in range(len(search_algorithm_names)):
                        search_algorithm = search_algorithm_name_to_func[search_algorithm_names[index]]

                        start_timer = perf_counter()
                        solution = search_algorithm(graph, start, goal, heuristics) if index == 0 else search_algorithm(graph, start, goal)
                        end_timer = perf_counter()

                        search_runtime[index] += (end_timer - start_timer)
                        search_solution_length[index] += len(solution)
                        search_solution_cost[index] += Search.get_path_cost(graph, solution)

        number_of_paths = (number_of_nodes *(number_of_nodes - 1)) // 2

        for index in range(len(search_algorithm_names)):
            search_runtime[index] /= (number_of_paths * number_of_trial / 1000) # convert to milli second
            search_solution_length[index] /= (number_of_paths * number_of_trial)


        return search_runtime, search_solution_length


    @staticmethod
    def __take_sample_nodes_from_graph(graph: Graph, number_of_samples = 10) -> List[str]:
        """select `number_of_samples` random nodes from the graph"""
        list_of_nodes = list(graph.get_nodes().keys())
        randomlist = sample(list_of_nodes, number_of_samples)

        return randomlist

