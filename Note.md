

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