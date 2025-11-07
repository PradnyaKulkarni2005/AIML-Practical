def aStarAlgo(start_node, stop_node):
    open_set = set([start_node])   # Nodes discovered but not yet visited
    closed_set = set()             # Already visited nodes
    g = {}                         # Cost from start node to current node
    parents = {}                   # Keeps track of the path

    # Initialize start node
    g[start_node] = 0
    parents[start_node] = start_node

    while open_set:
        n = None

        # Find node with lowest f(n) = g(n) + h(n)
        for v in open_set:
            if n is None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v

        if n is None:
            print('Path does not exist!')
            return None

        # If goal is reached, reconstruct path
        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            print('Path found: {}'.format(path))
            return path

        # For each neighbor of current node
        for (m, weight) in get_neighbors(n):
            # If neighbor not visited, add it
            if m not in open_set and m not in closed_set:
                open_set.add(m)
                parents[m] = n
                g[m] = g[n] + weight
            else:
                # If a shorter path is found, update it
                if g[m] > g[n] + weight:
                    g[m] = g[n] + weight
                    parents[m] = n

                    if m in closed_set:
                        closed_set.remove(m)
                        open_set.add(m)

        # Mark current node as visited
        open_set.remove(n)
        closed_set.add(n)

    print('Path does not exist!')
    return None


# Function to get neighbors
def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return []


# Heuristic function
def heuristic(n):
    heuristic_dist = {
        'A': 11,
        'B': 6,
        'C': 99,
        'D': 1,
        'E': 7,
        'G': 0
    }
    return heuristic_dist[n]


# Graph representation
Graph_nodes = {
    'A': [('B', 2), ('E', 3)],
    'B': [('C', 1), ('G', 9)],
    'C': [],
    'E': [('D', 6)],
    'D': [('G', 1)],
    'G': []
}


# Run A* Algorithm
aStarAlgo('A', 'G')


# Output:- Path found: ['A', 'E', 'D', 'G']




