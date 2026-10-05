"""Shortest paths in a weighted graph — Dijkstra's algorithm.

This is a fast, dependency-free implementation built on the binary heap in
:mod:`heapq`.  It is the standard choice for non-negative edge weights and
runs in

    time  : O((V + E) log V)
    space : O(V)

where ``V`` is the number of vertices and ``E`` the number of edges.

How the speed is obtained
-------------------------
* **Adjacency maps** — the graph is ``{node: {neighbour: weight}}``, so
  looking up the weight of an edge is O(1) and a node's neighbours are
  iterated in O(degree) without scanning every edge.
* **Lazy deletion instead of decrease-key** — ``heapq`` has no decrease-key
  operation.  When a shorter route to a node is found we simply push the new
  distance again; the older, larger entry is discarded when it is popped.
  This is faster in practice in Python than maintaining an indexed heap.
* **Settled nodes are never revisited** — once an entry is popped it is final,
  and stale duplicates are skipped in O(1).
* **Early exit** — when only one target is asked for, the search stops the
  moment that target is popped (its distance is then guaranteed final).
* **Only reachable nodes are stored** — distances are kept in a dict that
  starts with the source alone instead of pre-filling every node with
  ``inf``, which keeps memory and start-up cost proportional to the part of
  the graph actually explored.

Graph format
------------
``graph`` is a mapping of mappings::

    {
        "A": {"B": 4, "C": 2},   # edge A->B of weight 4, A->C of weight 2
        "B": {"C": 5, "D": 10},
        "C": {},
    }

Nodes may be any hashable object that is **not** ``None`` (``None`` is used
internally to mark the start of a path).  Weights must be non-negative
numbers; a negative weight raises :class:`ValueError`.  Weight validation
costs one pass over the edges and can be skipped with ``check_weights=False``
when the same graph is queried repeatedly.

Example
-------
>>> graph = {"A": {"B": 1, "C": 4},
...          "B": {"C": 2, "D": 5},
...          "C": {"D": 1},
...          "D": {}}
>>> shortest_distances(graph, "A")
{'A': 0, 'B': 1, 'C': 3, 'D': 4}
>>> path, distance = shortest_path(graph, "A", "D")
>>> path
['A', 'B', 'C', 'D']
>>> distance
4
>>> shortest_path(graph, "D", "A")
(None, inf)
"""

from __future__ import annotations

import heapq
import sys
from itertools import count
from time import perf_counter
from typing import Hashable, Iterable, Mapping, Optional, Sequence

__all__ = [
    "INFINITY",
    "build_graph",
    "reconstruct_path",
    "shortest_distances",
    "shortest_path",
]

# A node may be any hashable object that is not None.
Node = Hashable
# Weighted graph, mapping each node to {neighbour: weight}.
WeightedGraph = Mapping[Node, Mapping[Node, float]]

INFINITY = float("inf")

# Shared empty adjacency used for nodes that only appear as neighbours
# (sinks).  Reusing one object avoids allocating a dict per such node.
_NO_NEIGHBOURS: Mapping[Node, float] = {}


def shortest_distances(
    graph: WeightedGraph,
    source: Node,
    *,
    check_weights: bool = True,
) -> dict[Node, float]:
    """Return the shortest distance from ``source`` to every reachable node.

    Unreachable nodes are simply absent from the result.

    >>> shortest_distances({"s": {"t": 3}}, "s")
    {'s': 0, 't': 3}

    Args:
        graph: adjacency mapping ``{node: {neighbour: weight}}`` with
            non-negative weights.
        source: node to start from; must be a key of ``graph``.
        check_weights: set to ``False`` to skip the O(E) scan that rejects
            negative edge weights.  Only safe when the graph has already been
            validated (e.g. for an earlier query on the same graph).

    Returns:
        ``{node: distance}`` for ``source`` and every node reachable from it.

    Raises:
        KeyError: if ``source`` is not a key of ``graph``.
        ValueError: if a negative edge weight is found.
    """
    distances, _ = _search(graph, source, target=None, check_weights=check_weights)
    return distances


def shortest_path(
    graph: WeightedGraph,
    source: Node,
    target: Node,
    *,
    check_weights: bool = True,
) -> tuple[Optional[list[Node]], float]:
    """Return ``(path, distance)`` for the cheapest route ``source -> target``.

    The search stops as soon as ``target`` is reached, which can be much
    cheaper than computing every distance.  If ``target`` cannot be reached,
    ``(None, INFINITY)`` is returned.

    >>> graph = {"A": {"B": 1, "C": 10}, "B": {"C": 2}, "C": {}}
    >>> shortest_path(graph, "A", "C")
    (['A', 'B', 'C'], 3)
    >>> shortest_path(graph, "C", "A")
    (None, inf)

    Args:
        graph: adjacency mapping ``{node: {neighbour: weight}}``.
        source: node to start from; must be a key of ``graph``.
        target: node to reach.  ``target == source`` yields ``([source], 0)``.
        check_weights: see :func:`shortest_distances`.

    Returns:
        ``(path, distance)`` where ``path`` is the list of nodes along a
        cheapest route (``None`` if there is no route) and ``distance`` is its
        total weight (``INFINITY`` if there is no route).

    Raises:
        KeyError: if ``source`` is not a key of ``graph``.
        ValueError: if a negative edge weight is found.
    """
    distances, parents = _search(graph, source, target=target, check_weights=check_weights)
    if target not in distances:
        return None, INFINITY
    return reconstruct_path(parents, source, target), distances[target]


def reconstruct_path(
    parents: Mapping[Node, Optional[Node]],
    source: Node,
    target: Node,
) -> Optional[list[Node]]:
    """Rebuild the node sequence of a path from a ``parents`` mapping.

    ``parents`` maps every discovered node to its predecessor on a shortest
    route, with ``parents[source] is None``.  Returns ``None`` if ``target``
    was never discovered.

    >>> parents = {"A": None, "B": "A", "C": "B"}
    >>> reconstruct_path(parents, "A", "C")
    ['A', 'B', 'C']
    >>> reconstruct_path(parents, "A", "Z") is None
    True
    """
    if target not in parents:
        return None
    path: list[Node] = []
    node: Optional[Node] = target
    while node is not None:  # parents form a chain that ends at source
        path.append(node)
        node = parents[node]
    path.reverse()
    if path[0] != source:  # defensive: chain belongs to another source
        return None
    return path


def build_graph(
    edges: Iterable[Sequence],
    *,
    directed: bool = True,
    default_weight: float = 1,
) -> dict[Node, dict[Node, float]]:
    """Build an adjacency map from an iterable of ``(u, v[, weight])`` edges.

    Parallel edges keep their smallest weight, and every node mentioned
    becomes a key (sinks included) so the result can be passed straight to
    :func:`shortest_distances` or :func:`shortest_path`.

    >>> build_graph([("a", "b", 2), ("b", "c")])
    {'a': {'b': 2}, 'b': {'c': 1}, 'c': {}}
    >>> build_graph([(1, 2, 5)], directed=False)
    {1: {2: 5}, 2: {1: 5}}
    """
    graph: dict[Node, dict[Node, float]] = {}
    for edge in edges:
        u, v = edge[0], edge[1]
        weight = edge[2] if len(edge) > 2 else default_weight
        neighbours = graph.setdefault(u, {})
        if v not in neighbours or weight < neighbours[v]:
            neighbours[v] = weight
        graph.setdefault(v, {})
        if not directed:
            back = graph[v]
            if u not in back or weight < back[u]:
                back[u] = weight
    return graph


def _search(
    graph: WeightedGraph,
    source: Node,
    target: Optional[Node],
    check_weights: bool,
) -> tuple[dict[Node, float], dict[Node, Optional[Node]]]:
    """Core Dijkstra search returning ``(distances, parents)``.

    ``target`` may be ``None`` (explore everything reachable); otherwise the
    loop stops as soon as ``target`` is settled.
    """
    if source not in graph:
        raise KeyError(f"source node {source!r} is not in the graph")
    if check_weights:
        _validate_non_negative(graph)

    distances: dict[Node, float] = {source: 0}
    parents: dict[Node, Optional[Node]] = {source: None}

    # ``count()`` breaks ties between equal distances so that node labels
    # never need to be comparable with each other (e.g. mixing str and int).
    sequence = count()
    heap = [(0, next(sequence), source)]
    heappush = heapq.heappush
    heappop = heapq.heappop

    while heap:
        distance, _, node = heappop(heap)
        if distance > distances[node]:
            continue  # stale duplicate left behind by a later improvement
        if node == target:
            break  # settled: this distance is optimal, no need to look further

        for neighbour, weight in graph.get(node, _NO_NEIGHBOURS).items():
            candidate = distance + weight
            if candidate < distances.get(neighbour, INFINITY):
                distances[neighbour] = candidate
                parents[neighbour] = node
                heappush(heap, (candidate, next(sequence), neighbour))

    return distances, parents


def _validate_non_negative(graph: WeightedGraph) -> None:
    """Raise :class:`ValueError` if any edge in ``graph`` has weight < 0.

    The common (valid) case walks only the weight values; the node names are
    looked up again just to build a helpful message when an edge is rejected.
    """
    for weights in graph.values():
        for weight in weights.values():
            if weight < 0:
                raise ValueError(_negative_weight_message(graph))


def _negative_weight_message(graph: WeightedGraph) -> str:
    """Describe the first negative edge found in ``graph``."""
    for u, neighbours in graph.items():
        for v, weight in neighbours.items():
            if weight < 0:
                return (
                    f"negative edge weight {weight!r} on edge {u!r} -> {v!r}; "
                    "Dijkstra's algorithm requires non-negative weights"
                )
    return "negative edge weight found; Dijkstra requires non-negative weights"


def _demo() -> None:
    """Print the shortest routes for a small hand-made graph."""
    # A small town map, undirected.
    edges = [
        ("Home", "Cafe", 4),
        ("Home", "Library", 2),
        ("Cafe", "Library", 5),
        ("Cafe", "Gym", 10),
        ("Library", "Park", 3),
        ("Park", "Gym", 4),
        ("Gym", "Office", 11),
        ("Park", "Office", 8),
    ]
    graph = build_graph(edges, directed=False)

    print("Graph edges:", len(edges), "| nodes:", len(graph))
    distances = shortest_distances(graph, "Home", check_weights=False)
    for node in sorted(distances):
        print(f"  Home -> {node:<8} {distances[node]:>3}")

    path, distance = shortest_path(graph, "Home", "Office")
    print(f"\nShortest route Home -> Office: {' -> '.join(path)} (cost {distance})")


def _benchmark(nodes: int = 50_000, edges: int = 250_000, seed: int = 0) -> None:
    """Time full shortest-path computation on a random sparse graph."""
    import random

    rng = random.Random(seed)
    graph: dict[int, dict[int, float]] = {i: {} for i in range(nodes)}
    for _ in range(edges):
        u = rng.randrange(nodes)
        v = rng.randrange(nodes)
        graph[u][v] = rng.randint(1, 1_000)

    start = perf_counter()
    shortest_distances(graph, 0)  # includes validation
    with_validation = perf_counter() - start

    start = perf_counter()
    distances = shortest_distances(graph, 0, check_weights=False)
    without_validation = perf_counter() - start

    print(
        f"\nBenchmark: {nodes:,} nodes, {edges:,} edges — "
        f"{len(distances):,} reachable from node 0"
    )
    print(f"  with weight validation   : {with_validation * 1e3:8.1f} ms")
    print(f"  without weight validation: {without_validation * 1e3:8.1f} ms")


if __name__ == "__main__":
    _demo()
    if "--bench" in sys.argv:
        _benchmark()
