
from collections import deque
from romaniaCity import Romania
from queue import Queue, PriorityQueue
import heapq
from sys import maxsize
from graph import Graph, Node
from typing import List, Optional, Tuple
from math import radians, sqrt, sin, cos, atan2

class Search:
    def bfs(self, graph: Graph, start: str, goal:str):
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


    def dfs(self, graph: Graph, start: str, goal:str):
        explored = set()
        stack = [start]

        if start == goal:
            return [start]

        while stack:
            node = stack.pop()

            if node not in explored:
                explored.add(node)

                if node == goal:
                    return [node]

                neighbors = graph.search(node).get_neighbors()

                for neighbor in neighbors:
                    if neighbor.name not in explored:
                        path = self.dfs(graph, neighbor.name, goal)

                        if path:
                            path.insert(0, node)
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
                        heapq.heappush(heap, (cost + graph.search(node).get_weight(neighbor), neighbor.name, path))

        return None


    def ids(self, graph: Graph, start: str, goal:str):
        depth = 0

        while True:
            result = self.dls(graph, start, goal, depth)

            if result is not None:
                return result

            depth += 1


    def dls(self, graph, start, goal, depth):
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

    def get_path_cost(self, graph: Graph, path: List[str]) -> int:
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

        for node in graph.nodes.keys():
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

            # if the node has already been explored, skip it
            if explored[vertex]:
                continue

            # mark the node as explored
            explored[vertex] = True

            vertex_node: Node = graph.search(vertex)

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


    def a_star_search(self, graph: Graph, start: str, goal: str, coordinates: dict[str: Tuple[float, float]]) -> Optional[List[str]]:
        node_data = {}

        for node in graph.nodes:
            node_data[node] = {
                'g_cost': maxsize,
                'h_cost': self.heuristics(node, goal, coordinates),
                'prev': None
            }

        node_data[start]['g_cost'] = 0

        # create a priority queue and add the start node
        priority_queue = PriorityQueue()
        priority_queue.put((0, start))

        while not priority_queue.empty():
            # get the minimum cost node from the priority queue
            current_cost, vertex = priority_queue.get()

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

                # calculate the new cost estimate
                new_g_cost = current_cost + vertex_node.get_weight(neighbor_node)
                new_h_cost = self.heuristics(neighbor, goal, coordinates)
                new_f_cost = new_g_cost + new_h_cost

                # if the new cost estimate is better than the current estimate, update it
                if new_f_cost < node_data[neighbor]['g_cost'] + node_data[neighbor]['h_cost']:
                    node_data[neighbor]['g_cost'] = new_g_cost
                    node_data[neighbor]['h_cost'] = new_h_cost
                    node_data[neighbor]['prev'] = vertex

                    priority_queue.put((new_f_cost, neighbor))

        # there's no vald path to the goal from the start given
        return None


    def haversine_distance(self, coordinate_1, coordinate_2) -> int|float:
        latitude_1, longitude_1 = coordinate_1
        latitude_2, longitude_2 = coordinate_2
        
        # Convert latitudes and longitudes from degrees to radians
        latitude_1_rad, longitude_1_rad = radians(latitude_1), radians(longitude_1)
        latitude_2_rad, longitude_2_rad = radians(latitude_2), radians(longitude_2)

        # Haversine formula 
        distance_latitude = latitude_2_rad - latitude_1_rad 
        distance_longitude = longitude_2_rad - longitude_1_rad 
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


    def evaluateHeuristice(self):
        passed = 0
        total = 0
        for node1 in romania_coordinates:
            for node2 in romania_coordinates:
                if node1 == node2:
                    continue
                search_cost = self.get_path_cost(romania, self.a_star_search(romania, node1, node2, romania_coordinates))
                heuristic_cost = self.heuristics(node1, node2, romania_coordinates)
                if search_cost < heuristic_cost:
                    print(f'Test {passed}: {node1} => {node2}, \ndifference {heuristic_cost-search_cost}\n' )
                else:
                    passed += 1
                total += 1
        print(f'Passed : {passed}/{total}')
        print(f'Failed : {total -passed}/{total}')
        print(f'{round(passed/total*100, 2)} % Effective Heuristics')

romania = Romania().getCity()
romania_coordinates : dict[str, Tuple[float, float]] = {
    "Arad": (46.18656, 21.31227),
    "Bucharest": (44.42676, 26.10254),
    "Craiova": (44.31813, 23.80450),
    "Drobeta": (44.62524, 22.65608),
    "Eforie": (44.06562, 28.63361),
    "Fagaras": (45.84164, 24.97264),
    "Giurgiu": (43.90371, 25.96993),
    "Hirsova": (44.68935, 27.94566),
    "Iasi": (47.15845, 27.60144),
    "Lugoj": (45.69099, 21.90346),
    "Mehadia": (44.90411, 22.36452),
    "Neamt": (46.97587, 26.38188),
    "Oradea": (47.05788, 21.94140),
    "Pitesti": (44.85648, 24.86918),
    "Rimnicu Vilcea": (45.10000, 24.36667),
    "Sibiu": (45.79833, 24.12558),
    "Timisoara": (45.75972, 21.22361),
    "Urziceni": (44.71667, 26.63333),
    "Vaslui": (46.64069, 27.72765),
    "Zerind": (46.62251, 21.51742)
}



search = Search()

path = search.a_star_search(romania, "Arad", "Fagaras", romania_coordinates)
print(path)
print(search.get_path_cost(romania, path))

# search.evaluateHeuristice()