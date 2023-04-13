from typing import Dict, List, Mapping, Tuple
from undirectedGraph import Graph
from centrality import Centrality as center
from romaniaCity import Romania
import matplotlib.pyplot as plt

# compute the centrality of every Romanian city.
romania = Romania()
romania_graph: Graph = romania.get_city()
romania_coordinates = romania.get_coordinates()

betweenness_centrality = center().betweenness_centrality(romania_graph)
closeness_centrality = center().closeness_centrality(romania_graph)
degree_centrality = center().degree_centrality(romania_graph)
# eigenvector_centrality = center().eigenvector_centrality(romania_graph)
katz_centrality = center().katz_centrality(romania_graph)
pagerank_centrality = center().pagerank_centrality(romania_graph)

centrality_list : List[Tuple[Mapping,str]] = [(betweenness_centrality, "Betweenness"),\
                (closeness_centrality, "Closeness"),(degree_centrality, "Degree"),\
                # (eigenvector_centrality, "Eigenvector"),\
                      (katz_centrality, "Katz"),\
                (pagerank_centrality, "PageRank")]

x_labels = list(romania_graph.get_nodes())

for centrality in centrality_list:
    y_values = []

    centrality_dict, centrality_name = centrality

    for key in x_labels:
        y_values.append(centrality_dict[key])

    # plot the bar chart
    plt.bar(x_labels, y_values)
    plt.xticks(rotation=90)
    plt.xlabel('Cities') 
    plt.ylabel('Centrality')
    plt.title(centrality_name)
    plt.show()

    # create a sorted list of cities by centrality
    sorted_cities = sorted(x_labels, key=lambda x: centrality_dict[x], reverse=True)

    # create a table displaying the ranking of cities by centrality
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    table_data = [["Rank", "City", centrality_name + " Centrality"]]
    for i in range(len(sorted_cities)):
        table_data.append([i+1, sorted_cities[i], round(centrality_dict[sorted_cities[i]], 4)]) # type: ignore
    ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center') # type: ignore
    fig.tight_layout()
    plt.show()
