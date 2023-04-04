
from collections import deque
from Romania import Romania
from queue import Queue
import heapq
from sys import maxsize
from Graph import Graph, Node
from typing import List, Optional, Tuple
from queue import PriorityQueue


def bfs(graph, start, goal):
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



def dfs(graph, start, goal):
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
                    path = dfs(graph, neighbor.name, goal)

                    if path:
                        path.insert(0, node)
                        return path

    return None


def ucs(graph, start, goal):
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


def ids(graph, start, goal):
    depth = 0

    while True:
        result = dls(graph, start, goal, depth)

        if result is not None:
            return result

        depth += 1


def dls(graph, start, goal, depth):
    if depth == 0 and start == goal:
        return [start]

    if depth > 0:
        neighbors = graph.search(start).get_neighbors()

        for neighbor in neighbors:
            result = dls(graph, neighbor.name, goal, depth - 1)

            if result is not None:
                result.insert(0, start)
                return result

    return None

def get_path_cost(graph: Graph, path: List[str]) -> int:
    cost = 0

    for index in range(1, len(path)):
        cost+= graph.get_cost(path[index-1], path[index])

    return cost

def bidirectional_search(graph, start, goal):
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

            goal_path = bfs(graph, goal, start_node)

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

            start_path = bfs(graph, start, goal_node)

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

# def get_minimum_node(node_data: dict, graph: Graph, explored) -> Optional[str]:
#     min_ = None
#     for node in graph.nodes:
#         if not explored[node]:
#             min_ = node if not min_ or (min_ and node_data[min_]['cost']) > node_data[node]['cost'] else min_
#             # if not min_:
#                 # min_ = node
#             # elif node_data[min_]['cost'] > node_data[node]['cost']:
#                 # min_ = node

#     return min_

# def greedy(graph: Graph, start: str, goal: str):
#     # node_data = { node_name: [current_min_cost, previous] }
#     node_data = {}

#     # assign a maxSize of current cost estimate and a prevoius of None
#     explored = {}

#     for node in graph.nodes.keys():
#         node_data[node] = {'cost': maxsize, 'prev':None}
#         explored[node] = False

#     # assign zero cost for the start node
#     node_data[start]['cost'] = 0
#     # print(explored)

#     while not explored[goal]:

#         # done with 0(n) time, but can be done using a priority queue.
#         vertex = get_minimum_node(node_data, graph, explored)

#         explored[vertex] = True

#         vertex_node : Node = graph.search(vertex)

#         # get the neighbours of vertex
#         # update their estimate
#         for neighbor in vertex_node.get_neighbors():
#             neighbor = neighbor.name
#             neighbor_node = graph.search(neighbor)
#             if node_data[vertex]['cost'] + vertex_node.get_weight(neighbor_node) < node_data[neighbor]['cost']:
#                 node_data[neighbor]['cost'] = node_data[vertex]['cost'] + vertex_node.get_weight(neighbor_node)
#                 node_data[neighbor]['prev'] = vertex

#     path = deque()
#     path.append(goal)

#     while node_data[path[0]]['prev']:
#         path.appendleft(node_data[path[0]]['prev'])

#     return list(path)


def greedy_search(graph: Graph, start: str, goal: str):
    # node_data = { node_name: [current_min_cost, previous] }
    node_data = {}

    # assign a maxSize of current cost estimate and a prevoius of None
    explored = {}

    for node in graph.nodes.keys():
        node_data[node] = {'cost': maxsize, 'prev':None}
        explored[node] = False

    # assign zero cost for the start node
    node_data[start]['cost'] = 0

    # create a priority queue and add the start node
    priority_queue = [(0, start)]

    while priority_queue:
        # get the minimum cost node from the priority queue
        current_cost, vertex = heapq.heappop(priority_queue)

        # if the node has already been explored, skip it
        if explored[vertex]:
            continue

        # mark the node as explored
        explored[vertex] = True

        vertex_node : Node = graph.search(vertex)

        # if we have reached the goal node, stop the search
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
                heapq.heappush(priority_queue, (new_cost, neighbor))

    # constructing the path
    path = deque()
    path.append(goal)

    while node_data[path[0]]['prev']:
        path.appendleft(node_data[path[0]]['prev'])

    return list(path)


def a_start_search(graph: Graph, start: str, goal: str) -> Optional[List[str]]:
    pass


def heuristics(node, goal):
    return None

romania = Romania().getCity()

path =  a_start_search(romania, "Zerind", "Eforie")

