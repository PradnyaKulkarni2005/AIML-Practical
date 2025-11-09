
# 💧 Water Jug Problem — AI Implementation

## 📘 Introduction

The **Water Jug Problem** is a classic **Artificial Intelligence search problem** that demonstrates how **state-space search algorithms** (like BFS or DFS) can be used to solve real-world logical puzzles.

**Problem statement:**
Given two jugs with capacities **X** and **Y**, measure exactly **Z** liters of water using a series of valid operations.

---

## 🧩 Example

You have:

* Jug 1: 4 liters capacity
* Jug 2: 3 liters capacity
* Goal: Measure exactly 2 liters

---

## ⚙️ Algorithm Used

We use **Breadth-First Search (BFS)** to explore all possible states systematically.

Each **state** is represented as `(x, y)`
where

* `x` = amount of water in Jug1
* `y` = amount of water in Jug2

---

## 🧠 Operations Allowed

At any state `(x, y)`, you can perform the following actions:

1. **Fill Jug1 completely** → `(jug1_capacity, y)`
2. **Fill Jug2 completely** → `(x, jug2_capacity)`
3. **Empty Jug1** → `(0, y)`
4. **Empty Jug2** → `(x, 0)`
5. **Pour water from Jug1 → Jug2**
6. **Pour water from Jug2 → Jug1**

---

## 🧾 Python Code

```python
from collections import deque

def water_jug_problem(jug1_capacity, jug2_capacity, target):
    visited = set()
    queue = deque([(0, 0)])  # Start with both jugs empty

    while queue:
        x, y = queue.popleft()

        # Check goal
        if x == target or y == target:
            print(f"Solution found: Jug1 = {x}L, Jug2 = {y}L")
            return True

        if (x, y) in visited:
            continue

        visited.add((x, y))

        # Possible operations
        next_states = [
            (jug1_capacity, y),  # Fill Jug1
            (x, jug2_capacity),  # Fill Jug2
            (0, y),              # Empty Jug1
            (x, 0),              # Empty Jug2
            (x - min(x, jug2_capacity - y), y + min(x, jug2_capacity - y)),  # Pour Jug1→Jug2
            (x + min(y, jug1_capacity - x), y - min(y, jug1_capacity - x))   # Pour Jug2→Jug1
        ]

        for state in next_states:
            if state not in visited:
                queue.append(state)

    print("No solution found.")
    return False


# Example Run
water_jug_problem(4, 3, 2)
```

---

## 🧮 Step-by-Step Solution Example

| Step | Action                      | Jug1 (4L) | Jug2 (3L)          |
| ---- | --------------------------- | --------- | ------------------ |
| 1    | Fill Jug2                   | 0         | 3                  |
| 2    | Pour Jug2 → Jug1            | 3         | 0                  |
| 3    | Fill Jug2 again             | 3         | 3                  |
| 4    | Pour Jug2 → Jug1 until full | 4         | 2 ✅ (Goal reached) |

---

## 📊 Concepts Covered

| Concept                        | Description                                                                           |
| ------------------------------ | ------------------------------------------------------------------------------------- |
| **State Space Representation** | Every possible configuration of water in jugs forms a unique state `(x, y)`.          |
| **Search Algorithm (BFS)**     | Explores all possible moves level by level to find the shortest path to the solution. |
| **Transition Function**        | Defines how one state leads to another using allowed operations.                      |
| **Goal Test**                  | Check if `x == target` or `y == target`.                                              |
| **Visited States**             | Used to avoid revisiting already explored configurations.                             |

---

## 🧩 Advantages of BFS in Water Jug Problem

* Finds the **shortest sequence of actions**.
* Ensures all possibilities are explored systematically.
* Guarantees optimal solution (if one exists).

---

## 💬 Common Viva Questions & Answers

### 1️⃣ What is the Water Jug Problem?

It’s a classic AI problem used to demonstrate **state-space search** where the goal is to measure a specific quantity using jugs of fixed capacities.

### 2️⃣ Why do we use BFS for this problem?

Because BFS explores all states level-wise and ensures finding the **minimum number of steps** to reach the goal.

### 3️⃣ What are the possible operations in this problem?

Fill, Empty, or Pour between the two jugs.

### 4️⃣ How are states represented?

Each state is represented as a tuple `(x, y)` — the current amount of water in Jug1 and Jug2 respectively.

### 5️⃣ What is the role of the “visited” set?

It stores states that have already been explored to prevent **infinite loops** and redundant processing.

### 6️⃣ Can this problem be solved using DFS?

Yes, but DFS may not always give the shortest path and might go into deeper unnecessary states.

### 7️⃣ What is the time complexity of BFS here?

In worst case, it’s **O(m × n)** where `m` and `n` are the capacities of the two jugs.

### 8️⃣ What real-world problems can this represent?

Resource allocation, constraint satisfaction, and flow distribution problems in AI.

---

## 🧠 Key Takeaways

* The Water Jug Problem demonstrates **AI search techniques** in a clear, simple scenario.
* Using **BFS** ensures you find the most efficient way to reach the goal.
* Understanding **state transitions and goal testing** is key to solving such problems.

---

## 🏁 Output Example

```
Solution found: Jug1 = 4L, Jug2 = 2L
```

---

Would you like me to make this README into a **PDF file** (for submission) or a **Markdown (.md)** version for GitHub?
