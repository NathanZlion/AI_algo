

from typing import List, Mapping, Tuple
from centrality import Centrality as center
from romaniaCity import Romania
import matplotlib.pyplot as plt


# compute the centrality of every Romanian city.
romania_graph = Romania().get_city()
romania_coordinates = Romania().get_coordinates()

# eigenvector_centrality = center().eigenvector_centrality(romania_graph)
katz_centrality = center().katz_centrality(romania_graph)
# pagerank_centrality = center().pagerank_centrality(romania_graph)

centrality_list : List[Tuple[Mapping,str]] = [
        # (eigenvector_centrality, "Eigenvector"),\
        (katz_centrality, "Katz"),\
        # (pagerank_centrality, "PageRank")
    ]

x_values = list(romania_graph.get_nodes())

for centrality in centrality_list:
    y_values = []
    centrality_dict, centrality_name = centrality

    for key in x_values:
        y_values.append(centrality_dict[key])

    # create a figure with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(12,8))

    # plot the bar chart in the first column
    axs[0].bar(x_values, y_values)
    axs[0].set_xticklabels(x_values, rotation=90)
    axs[0].set_xlabel('Cities') 
    axs[0].set_ylabel('Centrality')
    axs[0].set_title(centrality_name)

    # create a sorted list of cities by centrality
    sorted_cities = sorted(x_values, key=lambda x: centrality_dict[x], reverse=True)

    # create a table displaying the ranking of cities by centrality in the second column
    table_data = [["Rank", "City", centrality_name + " Centrality"]]
    for i in range(len(sorted_cities)):
        table_data.append([str(i+1), sorted_cities[i], round(centrality_dict[sorted_cities[i]], 4)])

    axs[1].axis('off')
    axs[1].axis('tight')
    axs[1].table(cellText=table_data, colLabels=None, cellLoc='center', loc='center') # type: ignore

    fig.suptitle(f'{centrality_name} Centrality Graph and Table', fontsize=14)
    fig.tight_layout()

    # show the figure
    plt.show()
