from collections import deque

def bfs_shortest_path(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
    visited = [[False]*cols for _ in range(rows)]
    
    # Possible moves: up, down, left, right
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    # Queue holds (current_position, path_so_far)
    queue = deque([(start, [start])])
    visited[start[0]][start[1]] = True
    
    while queue:
        (x, y), path = queue.popleft()
        
        # If goal is reached, return the path
        if (x, y) == goal:
            return path
        
        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Check boundaries and if cell is free & not visited
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and maze[nx][ny] == 0:
                visited[nx][ny] = True
                queue.append(((nx, ny), path + [(nx, ny)]))
    
    return None  # No path found

# Example maze: 0 = free, 1 = wall
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0]
]

start = (0, 0)
goal = (3, 4)

path = bfs_shortest_path(maze, start, goal)

if path:
    print("Shortest path:", path)
    print("Number of steps:", len(path)-1)
else:
    print("No path found.")
