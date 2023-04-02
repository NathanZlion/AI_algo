
from Romania import Romania as romania
from queue import Queue
import heapq


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


rom = romania().getCity()

print(bfs(rom, "Zerind", "Craiova"))
