# Python
Resource of requirements for learning python package 

## Dijkstra's shortest path — `dijkstra.py`

Dependency-free implementation of Dijkstra's algorithm for weighted graphs
with non-negative edge weights. Runs in `O((V + E) log V)` time and `O(V)`
space using an adjacency map plus the binary heap from `heapq` (lazy deletion
instead of decrease-key, with early exit when only one target is needed).

```python
from dijkstra import build_graph, shortest_distances, shortest_path

graph = build_graph([
    ("A", "B", 1),
    ("A", "C", 4),
    ("B", "C", 2),
    ("C", "D", 1),
], directed=True)

shortest_distances(graph, "A")          # {'A': 0, 'B': 1, 'C': 3, 'D': 4}
shortest_path(graph, "A", "D")          # (['A', 'B', 'C', 'D'], 4)
```

* Negative weights are rejected with `ValueError`; pass
  `check_weights=False` to skip that scan for repeat queries on the same graph.
* Unreachable distances/targets are reported by omission / `(None, inf)`.
* Run the tests and the demo benchmark:

```bash
python -m unittest        # 26 tests, includes a randomized Bellman-Ford cross-check
python -m doctest dijkstra.py
python dijkstra.py --bench
```
