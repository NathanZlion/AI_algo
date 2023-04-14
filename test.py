
from matplotlib import pyplot as plt
from graphGenerator import Graph_Generator
from runSearchExperimentTests import Experiment


graph, coordinates = Graph_Generator.generate(40, 0.5)


search_algorithms = ["A_star", "breadth_first_search", "bidirectional_search", "dijkstra_search", \
                    "uniform_cost_search", "iterative_deepening_search", "depth_first_search", \
                    "greedy_search"]

search_runtime = [0.0 for _ in range(8)]
search_solution_length = [0.0 for _ in range(8)]

search_runtime, search_solution_length = Experiment.run(graph, coordinates, 10)

plt.plot(search_algorithms, search_runtime)
plt.xticks(rotation=90)
plt.xlabel('Search algorithms') 
plt.ylabel('avg runtime (ms)')
plt.title("Runtime Graph") 
plt.show()

plt.plot(search_algorithms, search_solution_length)
plt.xticks(rotation=90)
plt.xlabel('Search algorithms') 
plt.ylabel('Solution length') 
plt.title("Solution length Graph") 
plt.show()
