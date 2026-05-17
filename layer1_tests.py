"""
karachi_transport/layer1_tests.py
-----------------------------------
Layer 1 — Test Runner
Runs BFS, DFS, and UCS on 5 deliberately chosen origin-destination pairs.

Test cases:
  1. Short direct route             : Saddar → Garden
  2. Long cross-city route          : Orangi Town → Quaidabad
  3. Route through bottleneck       : Orangi Town → DHA Phase V
  4. BFS vs UCS find different paths: Saddar → Gulshan-e-Iqbal
  5. DFS performs badly             : Clifton → Surjani Town
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from graph import KarachiGraph
from layer1_uninformed import bfs, dfs, ucs, SearchResult


# ══════════════════════════════════════════════════════════════════════════════
# Test case definitions
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    {
        "id": 1,
        "label": "Short direct route",
        "start": "Saddar",
        "goal":  "Garden",
        "rationale": "Only 1 hop (10 min). All algorithms should agree instantly."
    },
    {
        "id": 2,
        "label": "Long cross-city route",
        "start": "Orangi Town",
        "goal":  "Quaidabad",
        "rationale": "Opposite ends of the city. Exposes how algorithms handle long multi-hop paths."
    },
    {
        "id": 3,
        "label": "Route through bottleneck",
        "start": "Orangi Town",
        "goal":  "DHA Phase V",
        "rationale": "Direct edge costs 65 min. UCS should prefer the longer multi-hop path via Lyari/Saddar."
    },
    {
        "id": 4,
        "label": "BFS and UCS find different paths",
        "start": "Saddar",
        "goal":  "Gulshan-e-Iqbal",
        "rationale": "BFS finds fewest hops; UCS finds minimum time. Edge weights vary so the optimal-hop "
                     "path is NOT the minimum-cost path."
    },
    {
        "id": 5,
        "label": "DFS performs badly",
        "start": "Clifton",
        "goal":  "Surjani Town",
        "rationale": "Goal is in the far north. DFS may dive south/east first, exploring many irrelevant "
                     "nodes before reaching the goal, and may return a very long suboptimal path."
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

def run_all(graph: KarachiGraph) -> list[dict]:
    """Run all 3 algorithms on all 5 test cases. Returns list of result dicts."""
    all_results = []
    for tc in TEST_CASES:
        row = {
            "id":       tc["id"],
            "label":    tc["label"],
            "start":    tc["start"],
            "goal":     tc["goal"],
            "rationale": tc["rationale"],
            "BFS":      bfs(graph, tc["start"], tc["goal"]),
            "DFS":      dfs(graph, tc["start"], tc["goal"]),
            "UCS":      ucs(graph, tc["start"], tc["goal"]),
        }
        all_results.append(row)
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# Pretty-print helpers
# ══════════════════════════════════════════════════════════════════════════════

COL_W = {
    "algo":     6,
    "found":    5,
    "cost":     6,
    "hops":     5,
    "expanded": 9,
    "frontier": 9,
    "path":     60,
}

def _fmt_result(r: SearchResult) -> str:
    path = " → ".join(r.path) if r.found else "NOT FOUND"
    found = "YES" if r.found else "NO"
    cost = str(r.cost) if r.found else "-"
    hops = str(len(r.path) - 1) if r.found else "-"
    return (f"  {r.algorithm:<{COL_W['algo']}} | "
            f"found={found:<{COL_W['found']}} | "
            f"cost={cost:>{COL_W['cost']}}m | "
            f"hops={hops:>{COL_W['hops']}} | "
            f"expanded={r.nodes_expanded:>{COL_W['expanded']}} | "
            f"frontier_max={r.max_frontier:>{COL_W['frontier']}} | "
            f"{path}")


def print_results_table(all_results: list[dict]):
    bar = "=" * 140
    thin = "-" * 140
    print(f"\n{bar}")
    print("  LAYER 1 RESULTS — Uninformed Search (BFS / DFS / UCS)")
    print(bar)
    for row in all_results:
        print(f"\n  Test {row['id']}: {row['label']}")
        print(f"  {row['start']} → {row['goal']}")
        print(f"  Rationale: {row['rationale']}")
        print(thin)
        for algo in ("BFS", "DFS", "UCS"):
            print(_fmt_result(row[algo]))
    print(f"\n{bar}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis
# ══════════════════════════════════════════════════════════════════════════════

def print_analysis(all_results: list[dict]):
    print("\n" + "=" * 80)
    print("  ANALYSIS")
    print("=" * 80)

    # Test 3: bottleneck
    t3 = all_results[2]
    bfs_r, ucs_r = t3["BFS"], t3["UCS"]
    print(f"\nTest 3 — Bottleneck ({t3['start']} → {t3['goal']}):")
    print(f"  BFS path cost : {bfs_r.cost} min  ({len(bfs_r.path)-1} hops)")
    print(f"  UCS path cost : {ucs_r.cost} min  ({len(ucs_r.path)-1} hops)")
    if ucs_r.cost < bfs_r.cost:
        saving = bfs_r.cost - ucs_r.cost
        print(f"  UCS saves {saving} min by avoiding the direct 65-min bottleneck edge.")
    elif ucs_r.cost == bfs_r.cost:
        print("  Both found the same cost path (direct edge is still cheapest here).")

    # Test 4: different paths
    t4 = all_results[3]
    bfs_r4, ucs_r4 = t4["BFS"], t4["UCS"]
    print(f"\nTest 4 — BFS vs UCS ({t4['start']} → {t4['goal']}):")
    print(f"  BFS path : {' → '.join(bfs_r4.path)}  (cost={bfs_r4.cost} min, {len(bfs_r4.path)-1} hops)")
    print(f"  UCS path : {' → '.join(ucs_r4.path)}  (cost={ucs_r4.cost} min, {len(ucs_r4.path)-1} hops)")
    if bfs_r4.path != ucs_r4.path:
        print(f"  Algorithms DISAGREE: BFS optimises hops; UCS optimises total time.")
    else:
        print(f"  Algorithms found same path (optimal hop count == optimal cost here).")

    # Test 5: DFS vs UCS
    t5 = all_results[4]
    dfs_r5, ucs_r5 = t5["DFS"], t5["UCS"]
    print(f"\nTest 5 — DFS penalty ({t5['start']} → {t5['goal']}):")
    print(f"  DFS path ({len(dfs_r5.path)-1} hops, {dfs_r5.cost} min): {' → '.join(dfs_r5.path)}")
    print(f"  UCS path ({len(ucs_r5.path)-1} hops, {ucs_r5.cost} min): {' → '.join(ucs_r5.path)}")
    print(f"  DFS expanded {dfs_r5.nodes_expanded} nodes vs UCS {ucs_r5.nodes_expanded}.")
    if dfs_r5.cost > ucs_r5.cost:
        print(f"  DFS path is {dfs_r5.cost - ucs_r5.cost} min LONGER than optimal — wasted effort confirmed.")

    # Overall expansion comparison
    print("\n  Nodes Expanded Summary (across all 5 tests):")
    print(f"  {'Test':<8} {'BFS':>8} {'DFS':>8} {'UCS':>8}")
    print(f"  {'-'*36}")
    for row in all_results:
        print(f"  {row['id']:<8} {row['BFS'].nodes_expanded:>8} "
              f"{row['DFS'].nodes_expanded:>8} {row['UCS'].nodes_expanded:>8}")

    print()
    print("  KEY OBSERVATIONS:")
    print("  1. BFS guarantees fewest hops but ignores edge weights — suboptimal in time.")
    print("  2. DFS is neither hop-optimal nor cost-optimal; it can return very long paths.")
    print("  3. UCS always finds the minimum-cost (minimum travel time) path.")
    print("  4. The Orangi → DHA bottleneck clearly separates BFS and UCS behaviour.")
    print("  5. DFS node expansion is unpredictable — can be best or worst case depending")
    print("     on whether the initial branch leads toward or away from the goal.\n")


# ══════════════════════════════════════════════════════════════════════════════
# Visualise frontier/expansion as ASCII bar chart
# ══════════════════════════════════════════════════════════════════════════════

def print_bar_chart(all_results: list[dict]):
    print("=" * 80)
    print("  NODES EXPANDED — ASCII Bar Chart")
    print("=" * 80)
    max_val = max(
        r.nodes_expanded
        for row in all_results
        for r in (row["BFS"], row["DFS"], row["UCS"])
    )
    bar_width = 40

    for row in all_results:
        print(f"\n  Test {row['id']}: {row['label']}")
        for algo in ("BFS", "DFS", "UCS"):
            r = row[algo]
            n = r.nodes_expanded
            filled = int(n / max_val * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"    {algo:3s} [{bar}] {n}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    g = KarachiGraph()
    g.summary()
    g.print_bottlenecks()

    results = run_all(g)
    print_results_table(results)
    print_bar_chart(results)
    print_analysis(results)
