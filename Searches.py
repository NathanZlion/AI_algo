
from collections import deque
from queue import Queue, PriorityQueue
import heapq
import random
from sys import maxsize
from graph_generator import Graph_Generator
from romania_city import Romania
from undirected_graph import Graph
from typing import Dict, List, Optional, Set, Tuple
from math import asin, atan2, radians, sqrt, sin, cos

class Search:
    """ Implemented different methods to search a graph."""

    @staticmethod
    def bfs(graph: Graph, start: str, goal:str):
        """
        BFS : `Breadth First Search` is a graph traversal algorithm that explores the vertices of a graph\
        in breadth-first order, meaning it visits all the neighbors of a vertex before moving on to their\
        neighbors. If there's no path between the nodes it returns ` an empty list, []`.
        """

        explored = set()
        queue : Queue[List[str]] = Queue()
        queue.put([start])

        if start == goal:return [start]

        while not queue.empty():
            path = queue.get()
            node = path[-1]

            if node not in explored:
                explored.add(node)
                neighbors = graph.get_node(node).get_neighbors()

                for neighbor in neighbors:
                    new_path = path.copy()
                    new_path.append(neighbor.name)
                    queue.put(new_path)

                    if neighbor.name == goal:
                        return new_path


        return []


    @staticmethod
    def dfs_iterative(graph: Graph, start: str, goal:str):
        """Implements a `Depth first search` implemented iteratively for a graph and returns the path between the start\
            and goal. Returns `[] empty list` if there is no valid path."""
        explored = set()
        stack : List[List[str]] = [[start]]

        if start == goal:
            return [start]

        while stack:
            path = stack.pop()
            node = path[-1]

            if node not in explored:
                neighbors = graph.get_node(node).get_neighbors()

                for neighbor in neighbors:

                    # avoid going back and forth 
                    if neighbor.name in path:
                        continue

                    new_path = path[:]
                    new_path.append(neighbor.name)
                    stack.append(new_path)

                    if neighbor.name == goal:
                        return new_path

                explored.add(node)

        return []


    @staticmethod
    def dfs_recursive(graph: Graph, start: str, goal:str, visited=None):
        """Implements a `Depth first search` implemented recursively for a graph and returns the path between the start\
            and goal. Returns `[] empty list` if there is no valid path."""
        if visited is None:
            visited = set()
        visited.add(start)

        if start == goal:
            return [start]

        for neighbor in graph.get_node(start).get_neighbors():
            if neighbor.name not in visited:
                path = Search.dfs_recursive(graph, neighbor.name, goal, visited)

                if path:
                    path.insert(0, start)
                    return path

        return []


    @staticmethod
    def dijkstra_search(graph: Graph, start: str, goal: str) -> List[str]:
        """
            Finds the shortest path between start and goal in the graph using the dijkstra \
            searching algorithm. It works for a weighted graph too.
        """
        distances = {node: float('inf') for node in graph.get_nodes().keys()}
        parent_map : Dict[str, Optional[str]] = {node: None for node in graph.get_nodes().keys()}
        distances[start] = 0
        priority_queue : list[tuple[int|float, str]]= [(0, start)]
        explored : Set[str] = set()

        while len(priority_queue) > 0:
            curr_dist, curr_vertex = heapq.heappop(priority_queue)

            if curr_vertex == goal:
                return Search.__trace_path(goal, parent_map)

            # if a shorter path is already been explored don't discover
            if curr_dist > distances[curr_vertex] or curr_vertex in explored:
                continue

            for neighbor, weight in graph.get_node(curr_vertex).get_neighbors().items():
                new_distance = curr_dist + weight

                if new_distance < distances[neighbor.name]:
                    distances[neighbor.name] = new_distance
                    parent_map[neighbor.name] = curr_vertex
                    heapq.heappush(priority_queue, (new_distance, neighbor.name))

            explored.add(curr_vertex)

        return []


    @staticmethod
    def ucs(graph: Graph, start: str, goal:str) -> List[str]:
        """
        UCS: `Uniform cost first search` for a graph and returns the path between the start\
        and goal. Returns `empty list` if there is no valid path. Explores all paths.
        """
        heap : List[Tuple[int|float, str, List[str]]] = [(0, start, [])]
        explored = set()

        while heap:
            # get the least cost path so far.
            (curr_cost, node, path) = heapq.heappop(heap)

            if node not in explored:
                explored.add(node)
                path = path + [node]

                if node == goal:
                    return path

                neighbors = graph.get_node(node).get_neighbors()

                for neighbor in neighbors:
                    if neighbor.name not in explored:
                        cost_of_path = curr_cost + graph.get_node(node).get_weight(neighbor)
                        heapq.heappush(heap, (cost_of_path, neighbor.name, path.copy()))

        return []


    @staticmethod
    def dls(graph: Graph, start: str, goal: str, max_depth: int = 20):
        """
        DLS : `Depth limited search` algorithm. Returns a list path to the goal if \
        the goal is found within `max_depth` steps form the start node. returns `None` \
        if a valid path is inexistent within the specified `max_depth`.
        """ 
    
        if max_depth == 0 and start == goal:
            return [start]

        if max_depth > 0:
            neighbors = graph.get_node(start).get_neighbors()

            for neighbor in neighbors:
                result : Optional[List[str]] = Search.dls(graph, neighbor.name, goal, max_depth - 1)

                # result found, add current node at frond and return to caller.
                if result:
                    result.insert(0, start)
                    return result

        return None


    @staticmethod
    def ids(graph: Graph, start: str, goal:str):
        """
        Uses `Iterative deepining searching` algorithm that tries depth limited search \
        for a certain depth. If goal is not found within that depth, It tries to go deeper \
        one more depth and search again.
        """

        for depth in range(len(graph)):
            result = Search.dls(graph, start, goal, depth)

            if result:
                return result

        return []


    @staticmethod
    def get_path_cost(graph: Graph, path: List[str]) -> int|float:
        """Returns the cost of the path followed."""

        return sum([graph.get_cost(path[index], path[index+1]) \
                    for index in range(len(path)-1)])


    @staticmethod
    def bidirectional_search(graph: Graph, start: str, goal: str) -> List[str]:
        """`Bidirectional search` searches for a path between start and goal. It does this by\
            starting its search from both ends. Once the two paths meet from opposite sides it \
            returns the path taken from start to end. Returns an empty list if no path exists."""

        start_queue = Queue()
        start_queue.put([start])

        goal_queue = Queue()
        goal_queue.put([goal])

        start_explored = set()
        goal_explored = set()

        while not start_queue.empty() and not goal_queue.empty():
            start_path = start_queue.get()
            start_node = start_path[-1]

            if start_node not in start_explored:
                start_explored.add(start_node)

                goal_path = Search.bfs(graph, goal, start_node)

                if goal_path is not None:
                    goal_path.reverse()
                    start_path.extend(goal_path[1:])
                    return start_path

                neighbors = graph.get_node(start_node).get_neighbors()

                for neighbor in neighbors:
                    if neighbor.name not in start_explored:
                        new_path = list(start_path)
                        new_path.append(neighbor.name)
                        start_queue.put(new_path)

            goal_path = goal_queue.get()
            goal_node = goal_path[-1]

            if goal_node not in goal_explored:
                goal_explored.add(goal_node)

                start_path = Search.bfs(graph, start, goal_node)

                if start_path is not None:
                    start_path.reverse()
                    goal_path.extend(start_path[1:])
                    return goal_path

                neighbors = graph.get_node(goal_node).get_neighbors()

                for neighbor in neighbors:
                    if neighbor.name not in goal_explored:
                        new_path = list(goal_path)
                        new_path.append(neighbor.name)
                        goal_queue.put(new_path)

        return []


    @staticmethod
    def greedy_search(graph: Graph, start: str, goal: str, coordinates: dict[str, Tuple[float, float]]):
        """returns the path between goal and start in the graph using the greedy search algorithm."""

        parent_map = {}
        explored = set()

        # create a priority queue and add the start node
        priority_queue = PriorityQueue()
        priority_queue.put((Search.__heuristics(start, goal, coordinates), start)) # I have to get the coordinates of cities

        while not priority_queue.empty():
            # get the minimum cost node from the priority queue
            curr_heuristics, curr_node = priority_queue.get()

            # mark the node as explored
            explored.add(curr_node)


            # reached our goal state
            if curr_node == goal:
                path = deque()
                while curr_node:
                    path.appendleft(curr_node)
                    curr_node = parent_map.get(curr_node)
                
                return list(path)

            for neighbor_node in graph.get_node(curr_node).get_neighbors():
                neighbor_name = neighbor_node.name
                if neighbor_name in explored:
                    continue

                parent_map[neighbor_name] = curr_node
                priority_queue.put((Search.__heuristics(neighbor_name, goal, coordinates), neighbor_name))

        return []

    @staticmethod
    def a_star_search(graph: Graph, start: str, goal: str, heuristics: Dict[str, int | float]) -> List[str]:
        """returns the path between goal and start in the graph using the a* search algorithm.
        uses the heuristic input given for heuristics.

        """

        explored: Set[str] = set()
        g_cost = {node: float('inf') for node in graph.get_nodes()}
        parent: dict[str, Optional[str]] = {node: None for node in graph.get_nodes()}
        g_cost[start] = 0

        # create a priority queue and add the start node
        priority_queue: List[Tuple[int|float, str]] = [(0, start)]
        heapq.heapify(priority_queue)

        while priority_queue:
            current_cost, vertex = heapq.heappop(priority_queue)

            if vertex in explored:
                continue

            explored.add(vertex)
            if vertex == goal:
                return Search.__trace_path(goal, parent)

            # update the cost estimates of the neighbors
            vertex_node = graph.get_node(vertex)
            for neighbor_node in vertex_node.get_neighbors():
                neighbor = neighbor_node.name
                new_g_cost = current_cost + graph.get_cost(vertex, neighbor)

                # if the new cost estimate is better than the current estimate, update it
                if new_g_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_g_cost
                    parent[neighbor] = vertex
                    heapq.heappush(priority_queue, (g_cost[neighbor] + 0.07*heuristics[neighbor], neighbor))

        return []


    @staticmethod
    def __haversine_distance(coordinate_1, coordinate_2) -> float:
        """returns the distance between 2 coordinates on the earth's surface in `miles`."""

        latitude_1, longitude_1 = coordinate_1
        latitude_2, longitude_2 = coordinate_2

        # Convert latitudes and longitudes from degrees to radians
        latitude_1_rad, longitude_1_rad = radians(latitude_1), radians(longitude_1)
        latitude_2_rad, longitude_2_rad = radians(latitude_2), radians(longitude_2)

        # Haversine formula 
        distance_latitude = abs(latitude_2_rad - latitude_1_rad )
        distance_longitude = abs(longitude_2_rad - longitude_1_rad)
        a = sin(distance_latitude / 2)**2 + cos(latitude_1_rad) * cos(latitude_2_rad) * sin(distance_longitude / 2)**2
        distance_on_sphere = 2 * atan2(sqrt(a), sqrt(1-a)) 

        # Earth's radius in kilometers
        RADIUS_OF_EARTH = 6371

        distance_in_km = RADIUS_OF_EARTH * distance_on_sphere

        MILES_IN_KILOMETER = 0.614
        distance_in_mile = distance_in_km * MILES_IN_KILOMETER

        return distance_in_mile


    @staticmethod
    def __heuristics(city1: str, city2: str, coordinates_dict: dict[str, Tuple[float, float]]) -> int|float:

        coordinate_1 = coordinates_dict[city1]
        coordinate_2 = coordinates_dict[city2]

        return Search.__haversine_distance(coordinate_1, coordinate_2)


    @staticmethod
    def __trace_path(goal, parent_map) -> List[str]:
        """Traces a path to the start from the goal state and returns a list of path taken."""
        path = deque([goal])

        while parent_map[path[0]]:
            path.appendleft(parent_map[path[0]])

        return list(path)


    @staticmethod
    def calculate_heuristics(graph: Graph, coordinates) -> Dict[str, int | float]:
        """calculates a heuristic for all nodes, to precalculate necessary heuristics."""

        heuristics: Dict[str, int | float] = {}

        for start in graph.get_nodes():
            for goal in graph.get_nodes():
                heuristics[start] = Search.__heuristics(start, goal, coordinates)

        return heuristics


    #########################
    # these are extra methods used to evaluate the above methods.
    #########################

    @staticmethod
    def is_consistent_heuristics(graph: Graph, coordinates: Dict[str, Tuple[float, float]]):
        """
        Evaluates the consistency of the heuristics, that it doesn't overestimate the cost of path 
        between all all nodes. That is all estimate heuristics are <= the distance of path on graph.
        """
        passed = 0
        total = 0

        for node1 in coordinates:
            for node2 in coordinates:
                if node1 == node2:
                    continue

                search_cost = Search.get_path_cost(graph, Search.dijkstra_search(graph, node1, node2))
                heuristic_cost = Search.__heuristics(node1, node2, coordinates) 
                if search_cost < heuristic_cost:
                    # pass
                    print(f'Test {total}: {node1} => {node2}, \ndifference {heuristic_cost-search_cost}\n' )
                else:
                    passed += 1
                total += 1
            
        print(f'Passed : {passed}/{total}')
        print(f'Failed : {total -passed}/{total}')
        print(f'{round(passed/total*100, 2)} %  Consistent Heuristics')


    @staticmethod
    def is_admissible_heuristics(graph: Graph, goal: str, coordinates: Dict[str, Tuple[float, float]]):
        """
        Evaluates the effectiveness of the heuristics, that it doesn't overestimate the cost of path between all all nodes.
        That is all estimate heuristics are <= the distance of path on graph.
        """
        passed = 0
        total = 0

        for node in coordinates:
            if node == goal:
                continue

            search_cost = Search.get_path_cost(graph, Search.dijkstra_search(graph, node, goal))
            heuristic_cost = Search.__heuristics(node, goal, coordinates) 

            if search_cost < heuristic_cost:  # pass
                print(f'Test {total}: {node} => {goal}, \ndifference {heuristic_cost-search_cost}\n' )
            else:
                passed += 1
            total += 1
            
        print(f'Passed : {passed}/{total}')
        print(f'Failed : {total -passed}/{total}')
        print(f'{round(passed/total*100, 2)} % Admissible Heuristics for the supplied goal node')


    @staticmethod
    def astar_evalution(graph, coordinates):
        """check if a star is really getting an optimal solution."""
        passed = 0
        total = 0
        heuristics = Search.calculate_heuristics(graph, coordinates)

        for node1 in coordinates:
            for node2 in coordinates:
                if node1 == node2:
                    continue

                dijkstra_search_cost = Search.get_path_cost(graph, Search.dijkstra_search(graph, node1, node2))
                a_star_search_cost = Search.get_path_cost(graph, Search.a_star_search(graph, node1, node2, heuristics))

                if dijkstra_search_cost < a_star_search_cost:
                    print(f'Test {total+1}: {node1} => {node2}, \n' )
                else:
                    passed += 1
                total += 1
            
        print(f'Passed : {passed}/{total}')
        print(f'Failed : {total -passed}/{total}')
        print(f'>>> {round(passed/total*100, 2)} % Effective Heuristics ___ ')



print(Search.greedy_search(Romania().get_city(), "Oradea", "Craiova", Romania().get_coordinates()))