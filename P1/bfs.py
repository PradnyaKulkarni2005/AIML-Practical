from collections import deque

def bfs(start_node, stop_node):
    visited = set()
    queue = deque([[start_node]])  # Store paths

    while queue:
        path = queue.popleft()      # Get first path
        node = path[-1]             # Last node in path

        if node in visited:
            continue

        # Goal found
        if node == stop_node:
            print("BFS Path found:", path)
            return path

        visited.add(node)

        # Explore neighbors
        for (neighbor, _) in Graph_nodes.get(node, []):
            new_path = list(path)
            new_path.append(neighbor)
            queue.append(new_path)

    print("Path does not exist!")
    return None
