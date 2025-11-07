def dfs(start_node, stop_node):
    visited = set()
    stack = [[start_node]]# Store paths

    while stack:
        path = stack.pop()# Get last path
        node = path[-1]

        if node in visited:
            continue

        # Goal found
        if node == stop_node:
            print("DFS Path found:", path)
            return path

        visited.add(node)

        # Explore neighbors
        for (neighbor, _) in Graph_nodes.get(node, []):
            new_path = list(path)
            new_path.append(neighbor)
            stack.append(new_path)

    print("Path does not exist!")
    return None


#Graph
Graph_nodes = {
    'A': [('B', 2), ('E', 3)],
    'B': [('C', 1), ('G', 9)],
    'C': [],
    'E': [('D', 6)],
    'D': [('G', 1)],
    'G': []
}

dfs('A', 'G')

# Output-DFS Path found: ['A', 'E', 'D', 'G']

