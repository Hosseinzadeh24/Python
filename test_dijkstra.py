"""Tests for :mod:`dijkstra` (run with ``python -m unittest``)."""

from __future__ import annotations

import random
import unittest

from dijkstra import (
    INFINITY,
    build_graph,
    reconstruct_path,
    shortest_distances,
    shortest_path,
)

GRAPH = {
    "A": {"B": 1, "C": 4},
    "B": {"C": 2, "D": 5},
    "C": {"D": 1, "E": 7},
    "D": {"E": 3},
    "E": {},
    "F": {"A": 2},  # not reachable from A
}


def bellman_ford(graph, source):
    """Reference implementation (works with any weights) used to verify."""
    distances = {source: 0}
    for _ in range(len(graph)):  # |V| rounds is more than enough
        updated = False
        for u, neighbours in graph.items():
            if u not in distances:
                continue
            base = distances[u]
            for v, weight in neighbours.items():
                candidate = base + weight
                if candidate < distances.get(v, INFINITY):
                    distances[v] = candidate
                    updated = True
        if not updated:
            break
    return distances


def path_cost(graph, path):
    """Total weight of a path; raises if an edge does not exist."""
    return sum(graph[u][v] for u, v in zip(path, path[1:]))


class TestShortestDistances(unittest.TestCase):
    def test_known_graph(self):
        self.assertEqual(
            shortest_distances(GRAPH, "A"),
            {"A": 0, "B": 1, "C": 3, "D": 4, "E": 7},
        )

    def test_unreachable_nodes_are_omitted(self):
        distances = shortest_distances(GRAPH, "A")
        self.assertNotIn("F", distances)  # F only has an edge *into* A
        self.assertEqual(distances["E"], 7)

    def test_branch_node_reaches_the_rest(self):
        distances = shortest_distances(GRAPH, "F")
        self.assertEqual(distances["A"], 2)
        self.assertEqual(distances["B"], 3)

    def test_source_missing_raises(self):
        with self.assertRaises(KeyError):
            shortest_distances(GRAPH, "Z")

    def test_single_node(self):
        self.assertEqual(shortest_distances({"x": {}}, "x"), {"x": 0})

    def test_zero_weight_cycle_terminates(self):
        graph = build_graph([(1, 2, 0), (2, 3, 0), (3, 1, 0)], directed=False)
        self.assertEqual(shortest_distances(graph, 1), {1: 0, 2: 0, 3: 0})

    def test_self_loop_is_ignored(self):
        graph = {"a": {"a": 5, "b": 2}, "b": {}}
        self.assertEqual(shortest_distances(graph, "a"), {"a": 0, "b": 2})

    def test_negative_weight_raises(self):
        graph = {"a": {"b": -1}, "b": {}}
        with self.assertRaises(ValueError):
            shortest_distances(graph, "a")

    def test_negative_weight_anywhere_is_rejected_up_front(self):
        # The bad edge is not reachable from "a", but validation still catches it.
        graph = {"a": {"b": 1}, "b": {}, "c": {"d": -3}, "d": {}}
        with self.assertRaises(ValueError):
            shortest_distances(graph, "a")

    def test_check_weights_false_skips_validation(self):
        graph = {"a": {"b": 1}, "b": {}}
        self.assertEqual(
            shortest_distances(graph, "a", check_weights=False), {"a": 0, "b": 1}
        )

    def test_node_appearing_only_as_neighbour(self):
        graph = {"a": {"b": 1}}  # "b" is not a key of graph
        self.assertEqual(shortest_distances(graph, "a"), {"a": 0, "b": 1})

    def test_incomparable_node_labels(self):
        # Equal tentative distances must not force a comparison between nodes:
        # heapq would raise TypeError comparing 1 with (2, 3) without a tiebreak.
        graph = {"s": {1: 5, (2, 3): 5, "t": 9}, 1: {"t": 2}, (2, 3): {"t": 1}}
        distances = shortest_distances(graph, "s")
        self.assertEqual(distances["t"], 6)


class TestShortestPath(unittest.TestCase):
    def test_direct_route(self):
        path, distance = shortest_path(GRAPH, "A", "D")
        self.assertEqual(path, ["A", "B", "C", "D"])
        self.assertEqual(distance, 4)
        self.assertEqual(path_cost(GRAPH, path), distance)

    def test_prefers_cheaper_indirect_route(self):
        # B -> D costs 5, but B -> C -> D costs 3.
        path, distance = shortest_path(GRAPH, "A", "D")
        self.assertNotIn("B", path[2:])
        self.assertEqual(distance, 4)

    def test_source_equals_target(self):
        path, distance = shortest_path(GRAPH, "A", "A")
        self.assertEqual(path, ["A"])
        self.assertEqual(distance, 0)

    def test_unreachable_target(self):
        path, distance = shortest_path(GRAPH, "A", "F")
        self.assertIsNone(path)
        self.assertEqual(distance, INFINITY)

    def test_sink_target(self):
        path, distance = shortest_path(GRAPH, "A", "E")
        self.assertEqual(path_cost(GRAPH, path), distance)
        self.assertEqual(path, ["A", "B", "C", "D", "E"])
        self.assertEqual(distance, 7)

    def test_missing_source_raises(self):
        with self.assertRaises(KeyError):
            shortest_path(GRAPH, "nope", "A")

    def test_negative_weight_raises(self):
        graph = {"a": {"b": 1}, "b": {"c": -2}, "c": {}}
        with self.assertRaises(ValueError):
            shortest_path(graph, "a", "c")


class TestReconstructPath(unittest.TestCase):
    def test_missing_target(self):
        self.assertIsNone(reconstruct_path({"A": None}, "A", "Z"))

    def test_foreign_chain_is_rejected(self):
        self.assertIsNone(reconstruct_path({"B": None, "C": "B"}, "A", "C"))


class TestBuildGraph(unittest.TestCase):
    def test_directed(self):
        graph = build_graph([("a", "b", 2), ("b", "c")])
        self.assertEqual(graph, {"a": {"b": 2}, "b": {"c": 1}, "c": {}})

    def test_undirected(self):
        graph = build_graph([(1, 2, 5)], directed=False)
        self.assertEqual(graph, {1: {2: 5}, 2: {1: 5}})

    def test_parallel_edges_keep_smallest_weight(self):
        graph = build_graph([("a", "b", 7), ("a", "b", 2), ("a", "b", 9)])
        self.assertEqual(graph["a"]["b"], 2)


class TestAgainstReference(unittest.TestCase):
    """Randomized cross-check of distances and paths against Bellman-Ford."""

    def _random_graph(self, rng, nodes=40, edges=160):
        graph = {i: {} for i in range(nodes)}
        for _ in range(edges):
            u = rng.randrange(nodes)
            v = rng.randrange(nodes)
            weight = rng.randint(0, 20)  # includes zero-weight edges
            if v not in graph[u] or weight < graph[u][v]:
                graph[u][v] = weight
        return graph

    def test_distances_match_bellman_ford(self):
        for seed in range(25):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                graph = self._random_graph(rng)
                expected = bellman_ford(graph, 0)
                self.assertEqual(shortest_distances(graph, 0), expected)

    def test_paths_are_valid_and_optimal(self):
        for seed in range(25):
            rng = random.Random(1000 + seed)
            graph = self._random_graph(rng, nodes=30, edges=120)
            reference = bellman_ford(graph, 0)
            for target in rng.sample(range(30), 10):
                with self.subTest(seed=seed, target=target):
                    path, distance = shortest_path(graph, 0, target)
                    if target not in reference:
                        self.assertIsNone(path)
                        self.assertEqual(distance, INFINITY)
                        continue
                    self.assertEqual(distance, reference[target])
                    self.assertEqual(path[0], 0)
                    self.assertEqual(path[-1], target)
                    self.assertEqual(path_cost(graph, path), distance)


if __name__ == "__main__":
    unittest.main(verbosity=2)
