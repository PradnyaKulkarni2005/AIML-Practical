
## 🧠 **Concept: Tower of Hanoi**

* Tower of Hanoi is a **classic recursion problem**.
* You have **3 rods** and **N disks** of different sizes.
* All disks start on the **first rod (source)** and must be moved to the **third rod (destination)**.
* **Rules:**

  1. You can move **only one disk** at a time.
  2. You can move only the **top disk** from any rod.
  3. A **larger disk cannot be placed on a smaller disk**.

---

## 💻 **Python Code**

```python
# Tower of Hanoi implementation in Python

def tower_of_hanoi(n, source, auxiliary, destination):
    """
    Function to solve Tower of Hanoi puzzle.
    n: Number of disks
    source: The starting peg
    auxiliary: The helper peg
    destination: The target peg
    """

    if n == 1:
        print(f"Move disk 1 from {source} ➡️ {destination}")
        return

    # Step 1: Move n-1 disks from source to auxiliary
    tower_of_hanoi(n-1, source, destination, auxiliary)

    # Step 2: Move the largest disk from source to destination
    print(f"Move disk {n} from {source} ➡️ {destination}")

    # Step 3: Move the n-1 disks from auxiliary to destination
    tower_of_hanoi(n-1, auxiliary, source, destination)


# Driver Code
n = int(input("Enter the number of disks: "))
print("\nSteps to solve Tower of Hanoi:\n")
tower_of_hanoi(n, 'A', 'B', 'C')

# Total number of moves = 2^n - 1
print(f"\nTotal moves required: {2**n - 1}")
```

---

## 📤 **Sample Output**

```
Enter the number of disks: 3

Steps to solve Tower of Hanoi:

Move disk 1 from A ➡️ C
Move disk 2 from A ➡️ B
Move disk 1 from C ➡️ B
Move disk 3 from A ➡️ C
Move disk 1 from B ➡️ A
Move disk 2 from B ➡️ C
Move disk 1 from A ➡️ C

Total moves required: 7
```

---

## 🧩 **Explanation of Code**

1. **Recursive Function:**

   * Moves `n-1` disks from **source → auxiliary**.
   * Moves the largest disk directly **source → destination**.
   * Moves `n-1` disks from **auxiliary → destination**.

2. **Base Case:**

   * When only one disk remains, move it directly.

3. **Recurrence Relation:**

   * Number of moves required:
     [
     T(n) = 2T(n-1) + 1
     ]
     Hence, **Total moves = 2ⁿ - 1**

---

## ⏱️ **Time and Space Complexity**

| Type                 | Complexity | Explanation                                         |
| -------------------- | ---------- | --------------------------------------------------- |
| **Time Complexity**  | O(2ⁿ)      | Each disk movement doubles for every new disk added |
| **Space Complexity** | O(n)       | Due to recursion stack                              |

---

## 🧾 **Viva Questions**

| Question                                | Answer                                                                         |
| --------------------------------------- | ------------------------------------------------------------------------------ |
| What is Tower of Hanoi?                 | A mathematical puzzle involving moving disks between pegs under certain rules. |
| How many total moves are needed?        | ( 2^n - 1 )                                                                    |
| What concept is used in Tower of Hanoi? | **Recursion**                                                                  |
| Why recursion?                          | Each step depends on solving a smaller version of the same problem.            |
| What are the three rods used for?       | Source, Auxiliary (Helper), and Destination.                                   |

---

