"""
karachi_transport/layer1_uninformed.py
---------------------------------------
Layer 1 — Uninformed Search Algorithms
Implements BFS, DFS (with cycle detection), and UCS.

All three share a common interface:
    search(graph, start, goal) -> SearchResult

SearchResult fields:
    algorithm       str
    start / goal    str
    path            list[str]  (empty if no path found)
    cost            int        (sum of edge weights along path; 0 if unweighted BFS)
    nodes_expanded  int        (number of nodes popped from frontier)
    max_frontier    int        (peak size of the frontier data structure)
    found           bool
"""

from __future__ import annotations
from collections import deque
import heapq
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
# Result container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    algorithm: str
    start: str
    goal: str
    path: list = field(default_factory=list)
    cost: int = 0
    nodes_expanded: int = 0
    max_frontier: int = 0
    found: bool = False
    heuristic_name: str = ""

    def __str__(self):
        if not self.found:
            return (f"[{self.algorithm:6s}] {self.start} → {self.goal} | "
                    f"NOT FOUND | expanded={self.nodes_expanded} | frontier_max={self.max_frontier}")
        path_str = " → ".join(self.path)
        return (f"[{self.algorithm:6s}] {self.start} → {self.goal} | "
                f"cost={self.cost:4d}m | hops={len(self.path)-1} | "
                f"expanded={self.nodes_expanded:4d} | frontier_max={self.max_frontier:4d} | "
                f"path: {path_str}")


# ══════════════════════════════════════════════════════════════════════════════
# Shared helper: reconstruct path from parent dict
# ══════════════════════════════════════════════════════════════════════════════

def _reconstruct_path(parent: dict, start: str, goal: str) -> list[str]:
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def _path_cost(graph, path: list[str]) -> int:
    """Sum edge weights along a path."""
    total = 0
    for i in range(len(path) - 1):
        total += graph.edge_weight(path[i], path[i + 1])
    return total


# ══════════════════════════════════════════════════════════════════════════════
# BFS — Breadth-First Search
# ══════════════════════════════════════════════════════════════════════════════

def bfs(graph, start: str, goal: str) -> SearchResult:
    """
    Breadth-First Search.
    Optimal for unweighted graphs (fewest hops). NOT optimal when edge weights differ.
    Frontier: FIFO queue. Explored set prevents re-visits.
    """
    result = SearchResult(algorithm="BFS", start=start, goal=goal)

    if start == goal:
        result.path = [start]
        result.found = True
        return result

    # Queue stores (node,)
    frontier = deque([start])
    explored = set()
    parent = {start: None}

    while frontier:
        result.max_frontier = max(result.max_frontier, len(frontier))
        node = frontier.popleft()

        if node in explored:
            continue
        explored.add(node)
        result.nodes_expanded += 1

        for neighbour, _ in graph.neighbours(node):
            if neighbour not in explored and neighbour not in parent:
                parent[neighbour] = node
                if neighbour == goal:
                    result.found = True
                    result.path = _reconstruct_path(parent, start, goal)
                    result.cost = _path_cost(graph, result.path)
                    result.max_frontier = max(result.max_frontier, len(frontier))
                    return result
                frontier.append(neighbour)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# DFS — Depth-First Search (with cycle detection)
# ══════════════════════════════════════════════════════════════════════════════

def dfs(graph, start: str, goal: str) -> SearchResult:
    """
    Depth-First Search with cycle detection via an explored set.
    NOT optimal — may find a very long path. Can perform badly when
    the goal is near but the chosen branch dives deep in the wrong direction.
    Frontier: LIFO stack.
    """
    result = SearchResult(algorithm="DFS", start=start, goal=goal)

    if start == goal:
        result.path = [start]
        result.found = True
        return result

    # Stack stores (node, path_so_far)
    # We track the full path on the stack so we can return it correctly.
    frontier = [(start, [start])]
    explored = set()

    while frontier:
        result.max_frontier = max(result.max_frontier, len(frontier))
        node, path = frontier.pop()

        if node in explored:
            continue
        explored.add(node)
        result.nodes_expanded += 1

        if node == goal:
            result.found = True
            result.path = path
            result.cost = _path_cost(graph, path)
            return result

        # Push neighbours in reverse alphabetical order so we expand
        # alphabetically first (gives deterministic behaviour)
        for neighbour, _ in reversed(graph.neighbours(node)):
            if neighbour not in explored:
                frontier.append((neighbour, path + [neighbour]))

    return result


# ══════════════════════════════════════════════════════════════════════════════
# UCS — Uniform Cost Search
# ══════════════════════════════════════════════════════════════════════════════

def ucs(graph, start: str, goal: str) -> SearchResult:
    """
    Uniform Cost Search (Dijkstra-style).
    Optimal for any non-negative edge weights. Expands nodes in order of
    cumulative path cost. Equivalent to Dijkstra's algorithm.
    Frontier: min-heap priority queue keyed on cumulative cost.
    """
    result = SearchResult(algorithm="UCS", start=start, goal=goal)

    if start == goal:
        result.path = [start]
        result.found = True
        return result

    # Heap entries: (cost, tie_break_counter, node, path)
    counter = 0
    frontier = [(0, counter, start, [start])]
    explored = set()
    best_cost = {start: 0}

    while frontier:
        result.max_frontier = max(result.max_frontier, len(frontier))
        cost, _, node, path = heapq.heappop(frontier)

        if node in explored:
            continue
        explored.add(node)
        result.nodes_expanded += 1

        if node == goal:
            result.found = True
            result.path = path
            result.cost = cost
            return result

        for neighbour, weight in graph.neighbours(node):
            new_cost = cost + weight
            if neighbour not in explored and new_cost < best_cost.get(neighbour, float('inf')):
                best_cost[neighbour] = new_cost
                counter += 1
                heapq.heappush(frontier, (new_cost, counter, neighbour, path + [neighbour]))

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Unified search dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def search(graph, start: str, goal: str, algorithm: str = "ucs") -> SearchResult:
    """
    Dispatch to BFS, DFS, or UCS by name.
    algorithm: 'bfs' | 'dfs' | 'ucs'
    """
    algo = algorithm.lower()
    if algo == "bfs":
        return bfs(graph, start, goal)
    elif algo == "dfs":
        return dfs(graph, start, goal)
    elif algo == "ucs":
        return ucs(graph, start, goal)
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose: bfs, dfs, ucs")
