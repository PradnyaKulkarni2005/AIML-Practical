# 📘 **Assignment 1 – Informed Search Algorithms**

### 🔍 **Topic:** Implement Informed Search Algorithms for Real-Life Problems

### 👩‍💻 **Algorithms Implemented:**

1. **BFS (Breadth First Search)**
2. **DFS (Depth First Search)**
3. **A* Algorithm**

---

## 🧠 1. Breadth First Search (BFS)

### **Concept:**

* BFS explores all the **neighboring nodes first** before moving to the next level.
* It uses a **Queue (FIFO)** data structure.
* It guarantees the **shortest path** in an **unweighted graph**.

### **Steps:**

1. Start from the source node and mark it as visited.
2. Add all neighbors of the current node to the queue.
3. Remove the front node from the queue and explore its neighbors.
4. Continue until you reach the goal or the queue is empty.

### **Code Summary:**

* Uses `deque` for queue operations.
* Stores entire paths to print the route from start to goal.
* Stops when the goal node is found.

### **Output Example:**

```
BFS Path found: ['A', 'B', 'G']
```

### **Time Complexity:**

* **O(V + E)**
  * V = number of vertices
  * E = number of edges

### **Space Complexity:**

* **O(V)** (due to storing visited nodes and queue)

### **Advantages:**

* Always finds the shortest path (in terms of number of edges).
* Good for unweighted graphs.

### **Disadvantages:**

* Uses a lot of memory due to queue.
* Slower for deep graphs.

---

## 🧩 2. Depth First Search (DFS)

### **Concept:**

* DFS explores as **deep as possible** along each branch before backtracking.
* It uses a **Stack (LIFO)** data structure.
* It may not always find the shortest path.

### **Steps:**

1. Start from the source node and push it to the stack.
2. Pop the top node and explore its first unvisited neighbor.
3. Continue going deeper until you reach the goal or a dead end.
4. Backtrack when no unvisited neighbor is found.

### **Code Summary:**

* Uses a stack to explore depth-first.
* Stores complete paths.
* Stops when the goal node is found.

### **Output Example:**

```
DFS Path found: ['A', 'E', 'D', 'G']
```

### **Time Complexity:**

* **O(V + E)**

### **Space Complexity:**

* **O(V)** (due to recursion/stack)

### **Advantages:**

* Uses less memory than BFS.
* Works well for searching deep nodes.

### **Disadvantages:**

* May get stuck in infinite loops if not handled properly.
* Does **not guarantee the shortest path**.

---

## 🚀 3. A* (A-Star) Algorithm

### **Concept:**

* A* is an **informed search algorithm** (uses knowledge about the problem).
* It uses a **heuristic function (h(n))** to estimate the cost from the current node to the goal.
* It also considers the **actual cost (g(n))** from the start node.
* Total cost function:
  [
  f(n) = g(n) + h(n)
  ]

### **Steps:**

1. Start from the start node.
2. Select the node with the **lowest f(n)**.
3. Update the cost of its neighbors.
4. Repeat until the goal node is found.

### **Code Summary:**

* `open_set`: nodes to be explored.
* `closed_set`: nodes already visited.
* `g[n]`: cost to reach node n.
* `heuristic(n)`: estimated cost from n to goal.
* Chooses the path with the smallest `f(n)`.

### **Output Example:**

```
Path found: ['A', 'E', 'D', 'G']
```

### **Time Complexity:**

* **Best Case:** O(E)
* **Average/Worst Case:** O(V + E) depending on the heuristic used.

### **Space Complexity:**

* **O(V)** (due to open and closed sets)

### **Advantages:**

* Finds the **shortest path efficiently**.
* Combines features of both **Uniform Cost Search** and **Greedy Best-First Search**.
* Very effective for **real-world pathfinding** (like maps and games).

### **Disadvantages:**

* Depends heavily on the **quality of heuristic**.
* Can consume more memory if the search space is large.

---

## 📊 **Comparison Table**

| Algorithm | Type       | Uses                       | Data Structure        | Optimal? | Complete? | Time Complexity | Space Complexity |
| --------- | ---------- | -------------------------- | --------------------- | -------- | --------- | --------------- | ---------------- |
| **BFS**   | Uninformed | Shortest path (unweighted) | Queue                 | ✅ Yes    | ✅ Yes     | O(V + E)        | O(V)             |
| **DFS**   | Uninformed | Deep search                | Stack                 | ❌ No     | ✅ Yes     | O(V + E)        | O(V)             |
| **A***    | Informed   | Shortest path (weighted)   | Priority Queue / Sets | ✅ Yes    | ✅ Yes     | O(V + E)        | O(V)             |

---

## 💡 **Key Viva Questions**

1. **What is the difference between BFS and DFS?**
   → BFS explores level by level; DFS goes deep first.

2. **Which algorithm guarantees the shortest path?**
   → BFS (for unweighted) and A* (for weighted graphs).

3. **What is a heuristic function in A*?**
   → It’s an estimate of the cost from the current node to the goal.

4. **Why is A* called an informed search?**
   → Because it uses heuristic knowledge to guide the search.

5. **When should you use BFS over DFS?**
   → Use BFS when the shortest path is required or the depth is unknown.

6. **What happens if the heuristic in A* is 0?**
   → It behaves like **Uniform Cost Search**.

---

## 🧾 **Summary**

| Algorithm | Ideal Use Case                                   |
| --------- | ------------------------------------------------ |
| **BFS**   | Finding the shortest path in an unweighted graph |
| **DFS**   | Searching deeper or exploring all paths          |
| **A***    | Finding the optimal path with cost and heuristic |

---

## 🛠️ **Real-Life Applications**

* **BFS:** GPS navigation in unweighted maps, social networks.
* **DFS:** Solving puzzles, maze problems, scheduling tasks.
* **A*:** Route planning (Google Maps, robotics, AI games).

---

## ✨ **Conclusion**

This assignment demonstrates how different search algorithms approach problem-solving:

* **BFS** and **DFS** explore blindly (uninformed).
* **A*** uses **knowledge (heuristics)** to make intelligent choices.

Each has its own **advantages**, **trade-offs**, and **use cases** depending on the problem.
