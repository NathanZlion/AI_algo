import heapq
from romaniaCity import Romania

def dijkstra(graph, start, end=None):
    # Initialize the distance and previous dictionaries
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}

    # Initialize the priority queue
    priority_queue = [(0, start)]
    
    while priority_queue:
        # Get the node with the minimum distance value
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # Skip already visited nodes
        if current_distance > distances[current_node]:
            continue
        
        # For each neighbor of the current node
        for neighbor, weight in graph[current_node].items():
            # Calculate the distance to the neighbor               
            new_distance = current_distance + weight
            
            # If the new distance is shorter, update the distances and previous dictionaries
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = current_node
                heapq.heappush(priority_queue, (new_distance, neighbor))
                
    # If the end node is specified, return the shortest path and its distance, otherwise return the distances dictionary
    if end:
        path = []
        node = end
        while node:
            path.append(node)
            node = previous[node]
        return distances[end], path[::-1]
    return distances

# Example usage
graph = {
    'A': {'B': 3, 'C': 4},
    'B': {'C': 1, 'D': 2, 'E': 6},
    'C': {'D': 4},
    'D': {'E': 1},
    'E': {}
}

rom_graph = {}

for node_name,node  in Romania().get_city().get_nodes().items():
    rom_graph[node_name] = {}
    neighbors = node.get_neighbors()
    for neighbor in neighbors:
        rom_graph[node_name][neighbor.name] = Romania().get_city().get_cost(node.name, neighbor.name)


shortest_path = dijkstra(rom_graph,"Arad", "Fagaras")
