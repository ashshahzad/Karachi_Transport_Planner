"""
karachi_transport/layer2_tests.py
-----------------------------------
Layer 2 — Heuristic Search Test Runner

Runs all algorithms (UCS from L1, Greedy/A*/IDA* with h1 and h2) on the same
5 origin-destination pairs used in Layer 1.

Outputs:
  1. Consistency check tables for h1 and h2
  2. Full comparison table (all algorithms × all test cases)
  3. ASCII bar chart — nodes expanded per algorithm per test
  4. Matplotlib bar chart saved as layer2_chart.png
  5. Written analysis: A* vs UCS, h2 vs h1, Greedy non-optimality example,
     IDA* vs A* comparison
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from graph import KarachiGraph
from layer1_uninformed import ucs
from layer1_uninformed import SearchResult as L1Result
from layer2_heuristic import (
    greedy, astar, idastar,
    h1_geographic, h2_karachi_aware,
    check_consistency, SearchResult
)

# ══════════════════════════════════════════════════════════════════════════════
# Same 5 test cases as Layer 1
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    {"id":1, "label":"Short direct route",          "start":"Saddar",       "goal":"Garden"},
    {"id":2, "label":"Long cross-city route",        "start":"Orangi Town",  "goal":"Quaidabad"},
    {"id":3, "label":"Bottleneck route",             "start":"Orangi Town",  "goal":"DHA Phase V"},
    {"id":4, "label":"BFS vs UCS divergence",        "start":"Saddar",       "goal":"Gulshan-e-Iqbal"},
    {"id":5, "label":"DFS worst-case",               "start":"Clifton",      "goal":"Surjani Town"},
]

# Edges to use for consistency checks (mix of short, long, bottleneck edges)
CONSISTENCY_EDGES = [
    ("Saddar",          "Garden"),
    ("Saddar",          "Clifton"),
    ("Orangi Town",     "DHA Phase V"),
    ("Orangi Town",     "North Nazimabad"),
    ("Clifton",         "DHA Phase V"),
    ("North Nazimabad", "Surjani Town"),
    ("Gulshan-e-Iqbal", "North Nazimabad"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Run all algorithms on all test cases
# ══════════════════════════════════════════════════════════════════════════════

# Algorithm label → (function, heuristic_fn, heuristic_name)
def run_all(graph):
    results = []
    for tc in TEST_CASES:
        s, g = tc["start"], tc["goal"]
        row = {
            "id":    tc["id"],
            "label": tc["label"],
            "start": s,
            "goal":  g,
            "UCS":        ucs(graph, s, g),
            "Greedy-h1":  greedy(graph, s, g, h1_geographic,   "h1"),
            "Greedy-h2":  greedy(graph, s, g, h2_karachi_aware, "h2"),
            "A*-h1":      astar(graph, s, g, h1_geographic,    "h1"),
            "A*-h2":      astar(graph, s, g, h2_karachi_aware,  "h2"),
            "IDA*-h1":    idastar(graph, s, g, h1_geographic,   "h1"),
            "IDA*-h2":    idastar(graph, s, g, h2_karachi_aware,"h2"),
        }
        results.append(row)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Print comparison table
# ══════════════════════════════════════════════════════════════════════════════

ALGO_KEYS = ["UCS", "Greedy-h1", "Greedy-h2", "A*-h1", "A*-h2", "IDA*-h1", "IDA*-h2"]

def print_comparison_table(results):
    bar = "=" * 160
    thin = "-" * 160
    print(f"\n{bar}")
    print("  LAYER 2 — Full Algorithm Comparison")
    print(bar)
    header = (f"  {'Algorithm':<14} | {'Cost(m)':>7} | {'Hops':>4} | "
              f"{'Expanded':>8} | {'Frontier':>8} | {'Optimal?':>8} | Path")
    for row in results:
        print(f"\n  Test {row['id']}: {row['label']}  ({row['start']} → {row['goal']})")
        print(thin)
        print(header)
        print(thin)
        ucs_cost = row["UCS"].cost
        for key in ALGO_KEYS:
            r = row[key]
            cost_str   = str(r.cost) if r.found else "—"
            hops_str   = str(len(r.path)-1) if r.found else "—"
            path_str   = " → ".join(r.path) if r.found else "NOT FOUND"
            optimal    = "✓" if r.found and r.cost == ucs_cost else ("✗" if r.found else "?")
            print(f"  {key:<14} | {cost_str:>7} | {hops_str:>4} | "
                  f"{r.nodes_expanded:>8} | {r.max_frontier:>8} | {optimal:>8} | {path_str}")
    print(f"\n{bar}\n")


# ══════════════════════════════════════════════════════════════════════════════
# ASCII bar chart — nodes expanded
# ══════════════════════════════════════════════════════════════════════════════

def print_ascii_chart(results):
    all_vals = [row[k].nodes_expanded for row in results for k in ALGO_KEYS]
    max_val = max(all_vals) or 1
    BAR = 35
    print("=" * 90)
    print("  NODES EXPANDED — ASCII Bar Chart")
    print("=" * 90)
    for row in results:
        print(f"\n  Test {row['id']}: {row['label']}  ({row['start']} → {row['goal']})")
        for key in ALGO_KEYS:
            n = row[key].nodes_expanded
            filled = int(n / max_val * BAR)
            bar = "█" * filled + "░" * (BAR - filled)
            print(f"    {key:<14} [{bar}] {n}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Matplotlib bar chart
# ══════════════════════════════════════════════════════════════════════════════

def save_matplotlib_chart(results):
    try:
        import matplotlib
        #matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        n_tests = len(results)
        n_algos = len(ALGO_KEYS)
        x = np.arange(n_tests)
        width = 0.11
        colors = ["#2E86AB", "#A23B72", "#C73E1D", "#3B1F2B",
                  "#44BBA4", "#E94F37", "#393E41"]

        fig, ax = plt.subplots(figsize=(16, 6))
        for i, (key, color) in enumerate(zip(ALGO_KEYS, colors)):
            vals = [row[key].nodes_expanded for row in results]
            offset = (i - n_algos/2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=key, color=color, alpha=0.88)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                            str(v), ha="center", va="bottom", fontsize=7, rotation=45)

        test_labels = [f"T{r['id']}: {r['label'][:18]}" for r in results]
        ax.set_xticks(x)
        ax.set_xticklabels(test_labels, fontsize=9)
        ax.set_ylabel("Nodes Expanded")
        ax.set_title("Layer 2 — Nodes Expanded by Algorithm and Test Case\n"
                     "Karachi Transport Planner | CS-5101", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "layer2_chart.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Chart saved → {out}\n")
        return out
    except ImportError:
        print("  (matplotlib not available — skipping PNG chart)\n")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Summary statistics table
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_stats(results):
    print("=" * 80)
    print("  SUMMARY STATISTICS — Totals across all 5 test cases")
    print("=" * 80)
    print(f"  {'Algorithm':<14} | {'Total Expanded':>14} | {'Total Cost':>10} | "
          f"{'Optimal?':>8} | {'Avg Expanded':>12}")
    print(f"  {'-'*70}")
    ucs_costs = [row["UCS"].cost for row in results]
    for key in ALGO_KEYS:
        total_exp  = sum(row[key].nodes_expanded for row in results)
        total_cost = sum(row[key].cost for row in results if row[key].found)
        optimal_count = sum(
            1 for row in results
            if row[key].found and row[key].cost == row["UCS"].cost
        )
        avg_exp = total_exp / len(results)
        opt_str = f"{optimal_count}/{len(results)}"
        print(f"  {key:<14} | {total_exp:>14} | {total_cost:>10} | "
              f"{opt_str:>8} | {avg_exp:>12.1f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Written analysis
# ══════════════════════════════════════════════════════════════════════════════

def print_analysis(results):
    print("=" * 80)
    print("  ANALYSIS")
    print("=" * 80)

    # 1. A* vs UCS improvement
    ucs_total  = sum(row["UCS"].nodes_expanded for row in results)
    astar_h1   = sum(row["A*-h1"].nodes_expanded for row in results)
    astar_h2   = sum(row["A*-h2"].nodes_expanded for row in results)
    pct_h1 = 100 * (ucs_total - astar_h1) / ucs_total if ucs_total else 0
    pct_h2 = 100 * (ucs_total - astar_h2) / ucs_total if ucs_total else 0
    print(f"\n1. A* vs UCS (nodes expanded, total across 5 tests):")
    print(f"   UCS        : {ucs_total} nodes")
    print(f"   A* (h1)    : {astar_h1} nodes  ({pct_h1:+.1f}% vs UCS)")
    print(f"   A* (h2)    : {astar_h2} nodes  ({pct_h2:+.1f}% vs UCS)")
    print(f"   On our small 20-node graph the gains are modest, but the pattern is")
    print(f"   consistent: h2 >= h1 dominance means A*-h2 expands fewer or equal nodes.")

    # 2. h2 dominates h1?
    h2_better = sum(
        1 for row in results
        if row["A*-h2"].nodes_expanded <= row["A*-h1"].nodes_expanded
    )
    print(f"\n2. Does h2 dominate h1 empirically?")
    print(f"   A*-h2 expanded <= A*-h1 in {h2_better}/{len(results)} test cases.")
    for row in results:
        n1 = row["A*-h1"].nodes_expanded
        n2 = row["A*-h2"].nodes_expanded
        tag = "h2 better" if n2 < n1 else ("tied" if n2 == n1 else "h1 better")
        print(f"   Test {row['id']}: A*-h1={n1}, A*-h2={n2}  → {tag}")

    # 3. Greedy non-optimality
    print(f"\n3. Did Greedy ever find a non-optimal path?")
    for row in results:
        ucs_cost = row["UCS"].cost
        for h in ("h1", "h2"):
            key = f"Greedy-{h}"
            gr = row[key]
            if gr.found and gr.cost != ucs_cost:
                diff = gr.cost - ucs_cost
                print(f"   YES — Test {row['id']} ({row['start']} → {row['goal']}) "
                      f"with {key}:")
                print(f"   Greedy path ({gr.cost} min): {' → '.join(gr.path)}")
                print(f"   Optimal     ({ucs_cost} min): {' → '.join(row['UCS'].path)}")
                print(f"   Greedy was {diff} min suboptimal (chose locally-promising "
                      f"direction but missed cheaper route).")
    # Check if Greedy was always optimal on this graph
    all_greedy_optimal = all(
        row[f"Greedy-{h}"].cost == row["UCS"].cost
        for row in results for h in ("h1","h2")
        if row[f"Greedy-{h}"].found
    )
    if all_greedy_optimal:
        print(f"   On this particular graph, Greedy happened to find optimal paths for")
        print(f"   all 5 test cases. This is a property of graph topology (few local")
        print(f"   optima traps), NOT a theoretical guarantee — Greedy has no optimality")
        print(f"   guarantee in general.")

    # 4. IDA* vs A*
    print(f"\n4. IDA* vs A* comparison:")
    print(f"   {'Test':<8} {'A*-h1 exp':>10} {'IDA*-h1 exp':>12} {'A*-h2 exp':>10} {'IDA*-h2 exp':>12}")
    print(f"   {'-'*55}")
    for row in results:
        print(f"   {row['id']:<8} "
              f"{row['A*-h1'].nodes_expanded:>10} "
              f"{row['IDA*-h1'].nodes_expanded:>12} "
              f"{row['A*-h2'].nodes_expanded:>10} "
              f"{row['IDA*-h2'].nodes_expanded:>12}")
    print()
    print(f"   IDA* uses O(depth) memory vs A*'s O(nodes) memory.")
    print(f"   The trade-off: IDA* may re-expand nodes across iterations, so its")
    print(f"   node-expansion count can exceed A*'s on dense graphs. On our sparse")
    print(f"    20-node graph the difference is small, making IDA* a good choice.")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    g = KarachiGraph()
    g.summary()

    # Consistency checks
    print("\n" + "="*80)
    print("  HEURISTIC CONSISTENCY CHECKS")
    print("="*80)
    check_consistency(g, h1_geographic,   "h1 (Geographic)", CONSISTENCY_EDGES)
    check_consistency(g, h2_karachi_aware, "h2 (Karachi-Aware)", CONSISTENCY_EDGES)

    # Run all algorithms
    results = run_all(g)

    print_comparison_table(results)
    print_ascii_chart(results)
    chart_path = save_matplotlib_chart(results)
    print_summary_stats(results)
    print_analysis(results)

    if chart_path:
        print(f"  PNG chart saved to: {chart_path}")
