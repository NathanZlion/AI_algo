
from collections import deque
import re
from romaniaCity import Romania
from queue import Queue, PriorityQueue
import heapq
from sys import maxsize
from undirectedGraph import Graph, Node
from typing import Dict, List, Optional, Tuple
from math import radians, sqrt, sin, cos, atan2

class Search:
    """Has different methods to search a graph."""

    def bfs(self, graph: Graph, start: str, goal:str):
        """using a `Queue` data structure, finds the shortest path between two nodes. If there's no path between the nodes it returns `None`"""
        explored = set()
        queue = Queue()

        queue.put([start])

        if start == goal:
            return [start]

        while not queue.empty():
            path = queue.get()
            node = path[-1]

            if node not in explored:
                neighbors = graph.search(node).get_neighbors()

                for neighbor in neighbors:
                    new_path = list(path)
                    new_path.append(neighbor.name)
                    queue.put(new_path)

                    if neighbor.name == goal:
                        return new_path

                explored.add(node)

        return None


    def dfs(self, graph: Graph, start: str, goal:str, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)

        if start == goal:
            return [start]

        for neighbor in graph.search(start).get_neighbors():
            if neighbor.name not in visited:
                path = self.dfs(graph, neighbor.name, goal, visited)

                if path is not None:
                    path.insert(0, start)
                    return path

        return None

    def ucs(self, graph: Graph, start: str, goal:str):
        heap = [(0, start, [])]
        explored = set()

        while heap:
            
            (cost, node, path) = heapq.heappop(heap)

            if node not in explored:
                explored.add(node)
                path = path + [node]

                if node == goal:
                    return path

                neighbors = graph.search(node).get_neighbors()

                for neighbor in neighbors:
                    if neighbor.name not in explored:
                        heapq.heappush(heap, (cost + graph.search(node).get_weight(neighbor), neighbor.name, path)) # type: ignore

        return None


    def ids(self, graph: Graph, start: str, goal:str):
        """Uses Iterative deepining searching algorithm to get path between the """
        depth = 0

        while True:
            result = self.dls(graph, start, goal, depth)

            if result is not None:
                return result

            depth += 1


    def dls(self, graph, start, goal, depth):
        """depth limited search algorithm."""
        if depth == 0 and start == goal:
            return [start]

        if depth > 0:
            neighbors = graph.search(start).get_neighbors()

            for neighbor in neighbors:
                result = self.dls(graph, neighbor.name, goal, depth - 1)

                if result is not None:
                    result.insert(0, start)
                    return result

        return None

    def get_path_cost(self, graph: Graph, path: List[str]) -> int|float:
        cost = 0

        for index in range(1, len(path)):
            cost+= graph.get_cost(path[index-1], path[index])

        return cost

    def bidirectional_search(self, graph: Graph, start: str, goal: str) -> Optional[List[str]]:
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

                goal_path = self.bfs(graph, goal, start_node)

                if goal_path is not None:
                    goal_path.reverse()
                    start_path.extend(goal_path[1:])
                    return start_path

                neighbors = graph.search(start_node).get_neighbors()

                for neighbor in neighbors:
                    if neighbor.name not in start_explored:
                        new_path = list(start_path)
                        new_path.append(neighbor.name)
                        start_queue.put(new_path)

            goal_path = goal_queue.get()
            goal_node = goal_path[-1]

            if goal_node not in goal_explored:
                goal_explored.add(goal_node)

                start_path = self.bfs(graph, start, goal_node)

                if start_path is not None:
                    start_path.reverse()
                    goal_path.extend(start_path[1:])
                    return goal_path

                neighbors = graph.search(goal_node).get_neighbors()

                for neighbor in neighbors:
                    if neighbor.name not in goal_explored:
                        new_path = list(goal_path)
                        new_path.append(neighbor.name)
                        goal_queue.put(new_path)

        return None


    def greedy_search(self, graph: Graph, start: str, goal: str):

        # node_data = { node_name: [current_min_cost, previous] }
        node_data = {}
        explored = {}

        for node in graph.get_nodes():
            node_data[node] = {'cost': maxsize, 'prev': None}
            explored[node] = False

        # assign zero cost for the start node
        node_data[start]['cost'] = 0

        # create a priority queue and add the start node
        priority_queue = PriorityQueue()
        priority_queue.put((0, start))

        while not priority_queue.empty():
            # get the minimum cost node from the priority queue
            current_cost, vertex = priority_queue.get()

            # mark the node as explored
            explored[vertex] = True

            vertex_node = graph.search(vertex)

            # if we have reached the goal node, stop the search immediately (greedily)
            if explored[goal]:
                break

            # update the cost estimates of the neighbors
            for neighbor in vertex_node.get_neighbors():
                neighbor = neighbor.name
                neighbor_node = graph.search(neighbor)

                # calculate the new cost estimate
                new_cost = current_cost + vertex_node.get_weight(neighbor_node)

                # if the new cost estimate is better than the current estimate, update it
                if new_cost < node_data[neighbor]['cost']:
                    node_data[neighbor]['cost'] = new_cost
                    node_data[neighbor]['prev'] = vertex

                    # add the neighbor to the priority queue
                    priority_queue.put((new_cost, neighbor))

        path = deque()
        path.append(goal)

        while node_data[path[0]]['prev']:
            path.appendleft(node_data[path[0]]['prev'])

        return list(path)


    def a_star_search(self, graph: Graph, start: str, goal: str, coordinates: dict[str: Tuple[float, float]]) -> List[str]: # type: ignore
        node_data = {}
        explored = set()

        for node in graph.get_nodes():
            node_data[node] = {
                'g_cost': float('inf'),
                'h_cost': self.heuristics(node, goal, coordinates),
                'prev': None
            }

        node_data[start]['g_cost'] = 0

        # create a priority queue and add the start node
        priority_queue = [(0, start)]

        while len(priority_queue) > 0:
            # get the minimum cost node from the priority queue
            current_cost, vertex = heapq.heappop(priority_queue)

            if vertex in explored:
                continue

            explored.add(vertex)

            # if we have reached the goal node, construct and return the path
            if vertex == goal:
                path = deque()
                while vertex is not None:
                    path.appendleft(vertex)
                    vertex = node_data[vertex]['prev']

                return list(path)

            # update the cost estimates of the neighbors
            vertex_node = graph.search(vertex)
            for neighbor_node in vertex_node.get_neighbors():
                neighbor = neighbor_node.name
                new_g_cost = current_cost + graph.get_cost(vertex, neighbor)

                # if the new cost estimate is better than the current estimate, update it
                if new_g_cost < node_data[neighbor]['g_cost']:
                    new_f_cost = new_g_cost + node_data[neighbor]['h_cost']
                    node_data[neighbor]['g_cost'] = new_g_cost
                    node_data[neighbor]['prev'] = vertex
                    heapq.heappush(priority_queue, (new_f_cost, neighbor))

        # there's no valid path to the goal from the start given
        return []

    def haversine_distance(self, coordinate_1, coordinate_2) -> float:
        latitude_1, longitude_1 = coordinate_1
        latitude_2, longitude_2 = coordinate_2

        # Convert latitudes and longitudes from degrees to radians
        latitude_1_rad, longitude_1_rad = radians(latitude_1), radians(longitude_1)
        latitude_2_rad, longitude_2_rad = radians(latitude_2), radians(longitude_2)

        # Haversine formula 
        distance_latitude = abs(latitude_2_rad - latitude_1_rad )
        distance_longitude = abs(longitude_2_rad - longitude_1_rad)
        a = sin(distance_latitude / 2)**2 + cos(latitude_1_rad) * cos(latitude_2_rad) * sin(distance_longitude / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a)) 

        # Earth's radius in kilometers
        R = 6371

        # Distance in kilometers
        distance = R * c

        return distance


    def heuristics(self, city1: str, city2: str, coordinates_dict: dict[str, float]) -> int|float:
        coordinate_1 = coordinates_dict[city1]
        coordinate_2 = coordinates_dict[city2]

        return self.haversine_distance(coordinate_1, coordinate_2)


    def evaluateHeuristice(self, graph: Graph, coordinates: Dict[str, Tuple[float, float]]):
        passed = 0
        total = 0
        for node1 in coordinates:
            for node2 in coordinates:
                if node1 == node2:
                    continue
                search_cost = self.get_path_cost(graph, self.a_star_search(graph, node1, node2, coordinates))  # type: ignore
                heuristic_cost = self.heuristics(node1, node2, coordinates)  # type: ignore
                if search_cost < heuristic_cost:
                    print(f'Test {passed}: {node1} => {node2}, \ndifference {heuristic_cost-search_cost}\n' )
                else:
                    passed += 1
                total += 1
        print(f'Passed : {passed}/{total}')
        print(f'Failed : {total -passed}/{total}')
        print(f'{round(passed/total*100, 2)} % Effective Heuristics')


    def graph_to_dict(self, graph: Graph):
        dictionary = {}

        for node_name,node  in graph.get_nodes().items():
            dictionary[node_name] = {}
            neighbors = node.get_neighbors()
            for neighbor in neighbors:
                dictionary[node_name][neighbor.name] = graph.get_cost(node.name, neighbor.name)
        
        return dictionary

    def dijkstra(self, graph: Graph, start:str, end:str):

        adj_dict = self.graph_to_dict(graph)

        # Initialize the distance and previous dictionaries
        distances = {node: float('inf') for node in adj_dict}
        distances[start] = 0
        previous = {node: None for node in adj_dict}

        # Initialize the priority queue
        priority_queue = [(0, start)]
        
        while priority_queue:
            # Get the node with the minimum distance value
            current_distance, current_node = heapq.heappop(priority_queue)
            
            # Skip already visited nodes
            if current_distance > distances[current_node]:
                continue
            
            # For each neighbor of the current node
            for neighbor, weight in adj_dict[current_node].items():
                # Calculate the distance to the neighbor               
                new_distance = current_distance + weight
                
                # If the new distance is shorter, update the distances and previous dictionaries
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_node # type: ignore
                    heapq.heappush(priority_queue, (new_distance, neighbor))
                    
        # If the end node is specified, return the shortest path and its distance, otherwise return the distances dictionary
        path = deque()
        node = end
        while node:
            path.appendleft(node)
            node = previous[node]
        return list(path)

romania = Romania().get_city()

search = Search()

path1 = search.ucs(romania, "Oradea", "Neamt")
print(path1, search.get_path_cost(romania, path1)) # type: ignore

path2 = search.dijkstra(romania, "Oradea", "Neamt")
print(path2, search.get_path_cost(romania, path2)) # type: ignore

path3 = search.a_star_search(romania, "Oradea", "Neamt", Romania().get_coordinates())
print(path3, search.get_path_cost(romania, path3)) # type: ignore

path4 = search.bidirectional_search(romania, "Oradea", "Neamt")
print(path4, search.get_path_cost(romania, path4)) # type: ignore

path5 = search.ids(romania, "Oradea", "Neamt")
print(path5, search.get_path_cost(romania, path5)) # type: ignore

path6 = search.bfs(romania, "Oradea", "Neamt")
print(path6, search.get_path_cost(romania, path6)) # type: ignore


path7 = search.greedy_search(romania, "Oradea", "Neamt")
print(path7, search.get_path_cost(romania, path7)) # type: ignore
