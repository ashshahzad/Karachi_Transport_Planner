"""
karachi_transport/graph.py
---------------------------
KarachiGraph — weighted undirected adjacency-list graph.
Supports building from EDGES, neighbour lookup, and pretty-printing.
"""

from collections import defaultdict
from graph_data import NODES, EDGES, COORDINATES, BOTTLENECKS


class KarachiGraph:
    """
    Weighted undirected graph of Karachi's transport network.

    Internally stored as:
        adjacency: dict[str, list[(neighbour: str, weight: int)]]
    """

    def __init__(self):
        self.adjacency: dict[str, list] = defaultdict(list)
        self.coordinates: dict[str, tuple] = dict(COORDINATES)
        self._build()

    def _build(self):
        for u, v, w in EDGES:
            self.adjacency[u].append((v, w))
            self.adjacency[v].append((u, w))
        # Sort neighbours alphabetically for deterministic BFS/DFS expansion order
        for node in self.adjacency:
            self.adjacency[node].sort(key=lambda x: x[0])

    # ── Core interface ─────────────────────────────────────────────────────

    def neighbours(self, node: str) -> list:
        """Return list of (neighbour, weight) for a given node."""
        return self.adjacency[node]

    def nodes(self) -> list:
        return list(self.adjacency.keys())

    def edge_weight(self, u: str, v: str) -> int | None:
        for neighbour, w in self.adjacency[u]:
            if neighbour == v:
                return w
        return None

    def num_nodes(self) -> int:
        return len(self.adjacency)

    def num_edges(self) -> int:
        total = sum(len(nbrs) for nbrs in self.adjacency.values())
        return total // 2   # undirected

    # ── Display helpers ────────────────────────────────────────────────────

    def summary(self):
        print(f"\n{'='*60}")
        print(f"  Karachi Transport Graph")
        print(f"  Nodes : {self.num_nodes()}")
        print(f"  Edges : {self.num_edges()}")
        print(f"{'='*60}")

    def print_adjacency(self):
        self.summary()
        for node in sorted(self.adjacency):
            nbrs = ", ".join(f"{v}({w}m)" for v, w in self.adjacency[node])
            print(f"  {node:<22} → {nbrs}")
        print()

    def print_bottlenecks(self):
        print("\n  Known bottleneck edges:")
        for u, v, note in BOTTLENECKS:
            w = self.edge_weight(u, v)
            print(f"    {u} ↔ {v}  [{w} min]  — {note}")
        print()


if __name__ == "__main__":
    g = KarachiGraph()
    g.print_adjacency()
    g.print_bottlenecks()
