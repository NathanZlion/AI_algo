# Introduction to AI - Assignment I

This repository contains code for the first assignment of the Introduction to AI course at Addis Ababa University. The objectives of the assignment include becoming comfortable with graphs and their operations, being able to implement graph search algorithms, benchmarking and comparing different search algorithms, and analyzing the results to deduce important findings.

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
- Depth-First Search (DFS)
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
- Install the required libraries using `pip install -r requirements.txt`
- Run the desired script using `python script_name.py`

___
The calculations explained.

Centralities:

___
- __Degree__ __Centrality__: refers to the number of connections a node has to the rest of the nodes.
```
formula:
degree_centrality(node) = number_of_edges(node) / n-1

n : total number of nodes in the graph.
```
___

- __Closeness__ __Centrality__: is the measure of average distance.

    - 1/ farness
    - 1 / sum of geodesic distances
    - 1 / sum length shortest path

___

- __Eigenvector__ __Centrality__ : Eigenvector Centrality is an algorithm that measures the transitive influence of nodes. Relationships originating from high-scoring nodes contribute more to the score of a node than connections from low-scoring nodes. A high eigenvector score means that a node is connected to many nodes who themselves have high scores.
`Steps`
    - Convert the network into a matrix.
        - Use an adjecency matrix to represent your graph
___

- __Betweenness__ __Centrality__: Betweenness centrality is a way of detecting the amount of influence a node has over the flow of information in a graph. It is often used to find nodes that serve as a bridge from one part of a graph to another. The algorithm calculates shortest paths between all pairs of nodes in a graph.

- `Formula`: 















___

## Author
- Nathnael Dereje
