
from collections import deque
from queue import Queue
import heapq
from undirected_graph import Graph
from typing import Dict, List, Optional, Set, Tuple
from math import atan2, radians, sqrt, sin, cos

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
    def get_path_cost(graph: Graph, path: List[str]) -> int|float:
        """Returns the cost of the path followed."""

        return sum([graph.get_cost(path[index], path[index+1]) \
                    for index in range(len(path)-1)])


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
    def __trace_path(goal, parent_map) -> List[str]:
        """Traces a path to the start from the goal state and returns a list of path taken."""
        path = deque([goal])

        while parent_map[path[0]]:
            path.appendleft(parent_map[path[0]])

        return list(path)


    @staticmethod
    def __heuristics(city1: str, city2: str, coordinates_dict: dict[str, Tuple[float, float]]) -> int|float:

        coordinate_1 = coordinates_dict[city1]
        coordinate_2 = coordinates_dict[city2]

        return Search.__haversine_distance(coordinate_1, coordinate_2)


    @staticmethod
    def calculate_heuristics(graph: Graph, coordinates) -> Dict[str, int | float]:
        """calculates a heuristic for all nodes, to precalculate necessary heuristics."""

        heuristics: Dict[str, int | float] = {}

        for start in graph.get_nodes():
            for goal in graph.get_nodes():
                heuristics[start] = Search.__heuristics(start, goal, coordinates)

        return heuristics
