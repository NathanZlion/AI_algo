# Introduction to AI - Assignment I

This repository contains code for the first assignment of the `Introduction to Artificial Intelligence` course at Addis Ababa University. The objectives of the assignment include becoming comfortable with graphs and their operations, being able to implement graph search algorithms, benchmarking and comparing different search algorithms, and analyzing the results to deduce important findings.

## Table of Contents

- [Graph Library](#graph-library)
- [Graph Search Algorithms](#graph-search-algorithms)
- [Centrality Indicators](#centrality-indicators)
- [Deliverables](#deliverables)

## Graph Library

The `graph_library` directory contains the implementation of a graph library with the following functionalities:

- Create a node
- Insert and delete edges and nodes
- Search for an item in a graph

The graph data presented on page 83 of the textbook is loaded using this self-made graph library.

## Graph Search Algorithms

The `graph_search_algorithms` directory contains implementations of the following search algorithms:

- Breadth-First Search (BFS)
- Depth-First Search (DFS) __ both recuersive and iterative approach.
- Uniform-Cost Search (UCS)
- Iterative Deepening Search
- Bidirectional Search
- Greedy Search
- A* Search

Using the graph from the `graph_library`, each algorithm is evaluated and benchmarked by finding the path between 10 randomly selected cities. Each experiment is run 10 times and the average time taken for each path search and the solution length are recorded. 

In addition, 16 random graphs with a number of nodes `n` = 10, 20, 30, 40 are created by randomly connecting nodes with the probability of edges `p` = 0.2, 0.4, 0.6, 0.8. For each graph setting, five nodes are randomly selected and the above search algorithms are used to find the paths between them. Each experiment is run 5 times and the average time taken in the five experiments is recorded. The average time and solution length on each graph size are plotted using `matplotlib.pyplot`.

## Centrality Indicators

The `centrality_indicators` directory contains implementations of the following centrality indicators:

- Degree Centrality
- Closeness Centrality
- Eigenvector Centrality
- Katz Centrality
- PageRank Centrality
- Betweenness Centrality

These centralities are computed for each node in the graph from the `graph_library`. The top-ranked cities in each centrality category are reported in a table and observations are summarized.

## Deliverables

The deliverables for this assignment include:

- A PDF file showing the results of the experiments and detailed observations, including tables, graphs, and plots highlighting the results.
- Organized code with proper naming.
- A `readme.txt` file listing the IDs of the group members.
- A zip file containing the code, the PDF file, and the `readme.txt` file, named `Group-x`, where `x` is the group ID.

Note: Definitions are excluded from the deliverables.

## How to Run the Code

- Clone this repository using `git clone https://github.com/NathanZlion/AI_algo.git`
- Navigate to the desired directory using `cd directory_name`
- Play with it
___
## Author
- Nathnael Dereje


- Dijkstra search, A* search (with a consistent heuristic), and bi-directional search (with a consistent heuristic) guarantee to get the best (shortest) path in an undirected weighted search.
