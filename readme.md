

## Introduction to AI - Assignment I

School of Information Technology & Engineering | Addis Ababa University
- Objectives of the assignment (30 points):
    - Be comfortable with graphs and their operations
    - Being able to implement graph search algorithms
    - Benchmark and compare different search algorithms
    -  Analyze the results and deduce important findings

* Make sure to write reusable code so that you don’t write the same algorithm multiple times
* You can use jupyter-notebook to write your algorithms
* Use matplot.pyplot library to plot the analysis


1. Develop a graph library. It should have the following functionalities
    1. Create a node
    2. Insert and delete edges and nodes
    3. Search for an item in a graph
    4. Using your self-made graph library, try loading the graph data presented on page 83rd of the textbook. The file containing the cities will be given to you. 


<img src=".\romania_city.png" alt=""></img>

2. Implement `BFS`, DFS, UCS, `Iterative Deepening`, `Bidirectional Search`, `Greedy`, and A* Search algorithms. Using the graph from Question 2, evaluate each of your algorithms and benchmark them.

    - The benchmark should be finding the path between each node. Randomly pick 10 cities. Find the path between them.

    - For each algorithm
        1. What is the average time taken for each path search?
        2. What is the solution length?

    - Each experiment should be run 10 times

    - Create random graphs with a number of nodes n = 10, 20, 30, 40.  Randomly connect nodes with the probability of edges p = 0.2, 0.4, 0.6, 0.8. In total, you will have 16 graphs. 

        1. Randomly select five nodes and apply the above algorithms to find paths between them in all 16 graph settings. 
        2. Register the time taken to find a solution for each algorithm and graph. Run each experiment 5 times and have the average of the time taken in the five experiments.
        3. Use matplotlib.pyplot to plot their average time and solution length on each graph sizes

3. In graph theory and network analysis, indicators of centrality assign numbers or rankings to nodes within a graph corresponding to their network position. For example, in a given social network, questions like who is an influencer? Who is the significant person? Who is the linkage between societies/groups? can easily be answered by calculating node centrality rankings.

    There are several centralities types. Compute the Degree, Closeness, Eigenvector, Katz, PageRank, and Betweenness centralities on the graph from Question 2. (You have to read online how to calculate these centralities). 

        1. Compute these centralities for each node
        2. Report a table containing top-ranked cities in each centrality category
        3. Summerise your observations

Deliverable: 
* A PDF file showing the results of your experiments and your detailed observations. Include tables, graphs, and plots highlighting the results. In your analysis, compare the results and explain why you saw the differences and the similarities. 
* Code organized with proper naming
* Have a readme.txt file you list the ids of the group members	
* Zip the code, the PDF file, and the readme.txt file in one folder and name it Group-x, where x is your group id.  

Don’t write definitions. Just the results.

How to submit it?
Submit it on Google Classroom. The deadline is April 8, midnight. 

#   A I _ a l g o  
 