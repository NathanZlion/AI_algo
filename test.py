# import networkx as nx

# # Define the list of edges
# romania_edges = [
#     ("Oradea", "Zerind", 71),
#     ("Oradea", "Sibiu", 151),
#     ("Zerind", "Arad", 75),
#     ("Sibiu", "Arad", 140),
#     ("Sibiu", "Fagaras", 99),
#     ("Sibiu", "Rimnicu Vilcea", 80),
#     ("Arad", "Timisoara", 118),
#     ("Timisoara", "Lugoj", 111),
#     ("Lugoj", "Mehadia", 70),
#     ("Mehadia", "Drobeta", 75),
#     ("Drobeta", "Craiova", 120),
#     ("Craiova", "Pitesti", 138),
#     ("Craiova", "Rimnicu Vilcea", 146),
#     ("Pitesti", "Rimnicu Vilcea", 97),
#     ("Pitesti", "Bucharest", 101),
#     ("Fagaras", "Bucharest", 211),
#     ("Bucharest", "Giurgiu", 90),
#     ("Bucharest", "Urziceni", 85),
#     ("Urziceni", "Vaslui", 142),
#     ("Urziceni", "Hirsova", 98),
#     ("Vaslui", "Iasi", 92),
#     ("Hirsova", "Eforie", 86),
#     ("Iasi", "Neamt", 87)
# ]

# # Create the graph and add the edges
# romania_graph = nx.Graph()
# romania_graph.add_weighted_edges_from(romania_edges)

# # Compute the eigenvector centrality of each node
# eigenvector_centralities = nx.eigenvector_centrality_numpy(romania_graph)

# # Print the centrality scores in descending order
# centralities_sorted = sorted(eigenvector_centralities.items(), key=lambda x: x[1], reverse=True)
# for city, centrality in centralities_sorted:
#     print(city, end=", ")


import networkx as nx

romania_graph = nx.Graph()
romania_edges = [
    ("Oradea", "Zerind", {"weight": 71}),
    ("Oradea", "Sibiu", {"weight": 151}),
    ("Zerind", "Arad", {"weight": 75}),
    ("Arad", "Sibiu", {"weight": 140}),
    ("Arad", "Timisoara", {"weight": 118}),
    ("Timisoara", "Lugoj", {"weight": 111}),
    ("Lugoj", "Mehadia", {"weight": 70}),
    ("Mehadia", "Drobeta", {"weight": 75}),
    ("Drobeta", "Craiova", {"weight": 120}),
    ("Craiova", "Pitesti", {"weight": 138}),
    ("Craiova", "Rimnicu Vilcea", {"weight": 146}),
    ("Rimnicu Vilcea", "Sibiu", {"weight": 80}),
    ("Sibiu", "Fagaras", {"weight": 99}),
    ("Fagaras", "Bucharest", {"weight": 211}),
    ("Pitesti", "Bucharest", {"weight": 101}),
    ("Rimnicu Vilcea", "Pitesti", {"weight": 97}),
    ("Urziceni", "Bucharest", {"weight": 85}),
    ("Urziceni", "Vaslui", {"weight": 142}),
    ("Vaslui", "Iasi", {"weight": 92}),
    ("Iasi", "Neamt", {"weight": 87}),
    ("Urziceni", "Hirsova", {"weight": 98}),
    ("Hirsova", "Eforie", {"weight": 86})
]

romania_graph.add_edges_from(romania_edges)

betweenness_centrality = nx.betweenness_centrality(romania_graph, weight="weight")
print(betweenness_centrality)


res = {'Oradea': 0.0, 'Zerind': 0.0, 'Sibiu': 0.21666666666666667, 'Arad': 0.0, 'Timisoara': 0.0, 'Lugoj': 0.0, 'Mehadia': 0.0, 'Drobeta': 0.0, 'Craiova': 0.0, 'Pitesti': 0.12222222222222222, 'Rimnicu Vilcea': 0.14444444444444443, 'Fagaras': 0.16666666666666666, 'Bucharest': 0.4333333333333333, 'Giurgiu': 0.0, 'Urziceni': 0.26666666666666666, 'Vaslui': 0.0, 'Iasi': 0.0, 'Neamt': 0.0, 'Hirsova': 0.0, 'Eforie': 0.0}
