"""
karachi_transport/layer3_optimization.py
-----------------------------------------
Layer 3 — Network Design via Local Search & Optimization

PROBLEM FORMULATION
===================
The Karachi Metropolitan Corporation (KMC) wants to select K active bus stops
from N=20 candidate locations and design routes connecting them, such that
the resulting network maximises coverage, connectivity, and efficiency.

State
-----
A NetworkState is a complete candidate solution:
    active_stops  : frozenset of K stop names currently selected
    active_edges  : frozenset of (u, v) tuples (u < v) — routes connecting stops

The state is complete: every evaluation needs the full configuration, not a
partial path. This is the defining feature of local search problems.

Objective Function
------------------
f(state) = α × coverage(state)
         + β × connectivity(state)
         + γ × efficiency(state)

Weights: α=0.40, β=0.30, γ=0.30

  coverage(state)
      Fraction of all 20 nodes that are either:
        (a) in the active stop set, OR
        (b) directly connected by one edge to an active stop in the full graph.
      Rationale: KMC's primary mandate is serving the most people. A stop
      within one hop covers its immediate neighbourhood.

  connectivity(state)
      Fraction of active-stop pairs that are reachable from each other
      through the active sub-network.
      = (connected pairs) / (K × (K-1) / 2)
      Penalises isolated clusters; rewards a single connected component.

  efficiency(state)
      Inverse of the normalised average shortest-path length between all
      reachable pairs of active stops (using only active edges).
      = 1 - (avg_shortest_path / max_possible_path)
      Rewards compact, fast networks. Unreachable pairs count as max cost.

Neighbourhood Function
----------------------
A neighbour of state s is any state reachable by ONE of:
    SWAP_STOP   : replace one active stop with one inactive stop
                  (keeps K constant; re-evaluates all edges incident to swapped stops)
    ADD_EDGE    : add one valid edge between two active stops (if not already present)
    REMOVE_EDGE : remove one existing active edge (if K>1 edges remain)

Design rationale:
  - SWAP_STOP explores the stop-selection space without changing network size.
  - ADD/REMOVE_EDGE explore the topology space independently.
  - Keeping all three move types gives a dense but tractable neighbourhood.
    A neighbourhood of ~K + E_active + E_possible moves per step is manageable
    for K=8 and N=20.

Algorithms
----------
1. Hill Climbing (steepest ascent) — with and without sideways moves
2. Random-Restart Hill Climbing — uses HC as a subroutine over R restarts
3. Simulated Annealing — configurable cooling schedule

All three share a common return type: OptimResult.
"""

from __future__ import annotations
import random
import math
import copy
from dataclasses import dataclass, field
from collections import deque
from itertools import combinations


# ══════════════════════════════════════════════════════════════════════════════
# Constants & weights
# ══════════════════════════════════════════════════════════════════════════════

ALPHA = 0.40   # weight for coverage
BETA  = 0.30   # weight for connectivity
GAMMA = 0.30   # weight for efficiency

DEFAULT_K = 8  # number of active bus stops to select


# ══════════════════════════════════════════════════════════════════════════════
# NetworkState
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NetworkState:
    """
    A complete candidate bus-network configuration.

    active_stops : frozenset[str]   — K selected stop names
    active_edges : frozenset[tuple] — selected (u,v) route edges (u<v)
    """
    active_stops: frozenset
    active_edges: frozenset

    def __hash__(self):
        return hash((self.active_stops, self.active_edges))

    def __eq__(self, other):
        return (self.active_stops == other.active_stops and
                self.active_edges == other.active_edges)

    def copy(self):
        return NetworkState(frozenset(self.active_stops),
                            frozenset(self.active_edges))

    def summary(self) -> str:
        stops = sorted(self.active_stops)
        edges = sorted(self.active_edges)
        return (f"  Stops ({len(stops)}): {', '.join(stops)}\n"
                f"  Edges ({len(edges)}): "
                + ", ".join(f"{u}↔{v}" for u, v in edges))


# ══════════════════════════════════════════════════════════════════════════════
# Objective function components
# ══════════════════════════════════════════════════════════════════════════════

def _bfs_reachable(start: str, adj: dict[str, list]) -> set:
    """BFS from start using adjacency dict. Returns all reachable nodes."""
    visited = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nb in adj.get(node, []):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return visited


def _build_active_adj(state: NetworkState) -> dict[str, list]:
    """Adjacency list restricted to active stops and active edges."""
    adj: dict[str, list] = {s: [] for s in state.active_stops}
    for u, v in state.active_edges:
        if u in state.active_stops and v in state.active_stops:
            adj[u].append(v)
            adj[v].append(u)
    return adj


def _bfs_distances(start: str, adj: dict[str, list]) -> dict[str, int]:
    """BFS shortest-hop distances from start (within active subgraph)."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nb in adj.get(node, []):
            if nb not in dist:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return dist


def coverage(state: NetworkState, all_nodes: list, full_adj: dict) -> float:
    """
    Fraction of all N nodes that are active stops OR directly adjacent to one.
    """
    covered = set(state.active_stops)
    for stop in state.active_stops:
        for nb, _ in full_adj.get(stop, []):
            covered.add(nb)
    return len(covered) / len(all_nodes)


def connectivity(state: NetworkState) -> float:
    """
    Fraction of active-stop pairs reachable from each other through
    the active sub-network.
    Returns 1.0 if K <= 1 (trivially connected).
    """
    stops = list(state.active_stops)
    K = len(stops)
    if K <= 1:
        return 1.0

    adj = _build_active_adj(state)
    total_pairs = K * (K - 1) / 2
    connected_pairs = 0

    for i, s in enumerate(stops):
        reachable = _bfs_reachable(s, adj)
        for j in range(i + 1, K):
            if stops[j] in reachable:
                connected_pairs += 1

    return connected_pairs / total_pairs


def efficiency(state: NetworkState) -> float:
    """
    1 - (avg_shortest_path / max_possible_path).
    Unreachable pairs are assigned max_possible_path (= K-1 hops).
    Returns 1.0 if K <= 1.
    """
    stops = list(state.active_stops)
    K = len(stops)
    if K <= 1:
        return 1.0

    adj = _build_active_adj(state)
    max_path = K - 1   # worst case: linear chain
    total_dist = 0
    pair_count = 0

    for s in stops:
        dists = _bfs_distances(s, adj)
        for t in stops:
            if t != s:
                total_dist += dists.get(t, max_path)
                pair_count += 1

    if pair_count == 0:
        return 0.0
    avg_path = total_dist / pair_count
    return 1.0 - (avg_path / max_path)


def objective(state: NetworkState, all_nodes: list, full_adj: dict) -> float:
    """
    f = α×coverage + β×connectivity + γ×efficiency
    Returns a value in [0, 1]. Higher is better.
    """
    cov  = coverage(state, all_nodes, full_adj)
    conn = connectivity(state)
    eff  = efficiency(state)
    return ALPHA * cov + BETA * conn + GAMMA * eff


# ══════════════════════════════════════════════════════════════════════════════
# Neighbourhood generation
# ══════════════════════════════════════════════════════════════════════════════

def _valid_edges_for_stops(stops: frozenset, full_adj: dict) -> set:
    """All edges in the full graph whose both endpoints are in stops."""
    valid = set()
    for u in stops:
        for v, _ in full_adj.get(u, []):
            if v in stops:
                edge = (min(u, v), max(u, v))
                valid.add(edge)
    return valid


def get_neighbours(state: NetworkState,
                   all_nodes: list,
                   full_adj: dict,
                   k: int) -> list[NetworkState]:
    """
    Generate all neighbours of state via three move types:

    1. SWAP_STOP   — replace one active stop with one inactive stop.
       After swap, rebuild active edges to keep only those valid for new stop set.
    2. ADD_EDGE    — add one valid edge between two active stops.
    3. REMOVE_EDGE — remove one active edge (at least one edge must remain).

    Returns a list of NetworkState objects.
    """
    neighbours = []
    inactive = [n for n in all_nodes if n not in state.active_stops]

    # ── Move 1: SWAP_STOP ────────────────────────────────────────────────────
    for stop_out in state.active_stops:
        for stop_in in inactive:
            new_stops = (state.active_stops - {stop_out}) | {stop_in}
            # Keep only edges that are still valid after the swap
            valid = _valid_edges_for_stops(new_stops, full_adj)
            new_edges = state.active_edges & valid
            # Ensure at least minimal connectivity: add edges incident to stop_in
            for v, _ in full_adj.get(stop_in, []):
                if v in new_stops:
                    edge = (min(stop_in, v), max(stop_in, v))
                    new_edges = new_edges | {edge}
            neighbours.append(NetworkState(frozenset(new_stops),
                                           frozenset(new_edges)))

    # ── Move 2: ADD_EDGE ─────────────────────────────────────────────────────
    valid_all = _valid_edges_for_stops(state.active_stops, full_adj)
    addable = valid_all - state.active_edges
    for edge in addable:
        neighbours.append(NetworkState(state.active_stops,
                                       state.active_edges | {edge}))

    # ── Move 3: REMOVE_EDGE ──────────────────────────────────────────────────
    if len(state.active_edges) > 1:
        for edge in state.active_edges:
            neighbours.append(NetworkState(state.active_stops,
                                           state.active_edges - {edge}))

    return neighbours


def random_neighbour(state: NetworkState,
                     all_nodes: list,
                     full_adj: dict,
                     k: int) -> NetworkState:
    """
    Pick one random neighbour (for SA and RRHC).
    Randomly selects a move type, then a random move of that type.
    Faster than enumerating all neighbours when only one is needed.
    """
    inactive = [n for n in all_nodes if n not in state.active_stops]
    valid_all = _valid_edges_for_stops(state.active_stops, full_adj)
    addable   = list(valid_all - state.active_edges)
    removable = list(state.active_edges) if len(state.active_edges) > 1 else []

    # Build list of available move types
    move_types = ["swap"]
    if addable:
        move_types.append("add")
    if removable:
        move_types.append("remove")

    move = random.choice(move_types)

    if move == "swap" and inactive:
        stop_out = random.choice(list(state.active_stops))
        stop_in  = random.choice(inactive)
        new_stops = (state.active_stops - {stop_out}) | {stop_in}
        valid = _valid_edges_for_stops(new_stops, full_adj)
        new_edges = state.active_edges & valid
        for v, _ in full_adj.get(stop_in, []):
            if v in new_stops:
                edge = (min(stop_in, v), max(stop_in, v))
                new_edges = new_edges | {edge}
        return NetworkState(frozenset(new_stops), frozenset(new_edges))

    elif move == "add":
        edge = random.choice(addable)
        return NetworkState(state.active_stops, state.active_edges | {edge})

    else:  # remove
        edge = random.choice(removable)
        return NetworkState(state.active_stops, state.active_edges - {edge})


# ══════════════════════════════════════════════════════════════════════════════
# Random initial state generator
# ══════════════════════════════════════════════════════════════════════════════

def random_state(all_nodes: list, full_adj: dict, k: int,
                 rng: random.Random | None = None) -> NetworkState:
    """
    Generate a random valid NetworkState:
      - Select K stops at random.
      - Add all edges valid for those stops (to maximise initial connectivity).
    """
    if rng is None:
        rng = random
    stops = frozenset(rng.sample(all_nodes, k))
    edges = frozenset(_valid_edges_for_stops(stops, full_adj))
    return NetworkState(stops, edges)


# ══════════════════════════════════════════════════════════════════════════════
# Result container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptimResult:
    algorithm:      str
    best_state:     NetworkState
    best_score:     float
    initial_score:  float
    steps:          int
    restarts:       int = 0
    score_history:  list = field(default_factory=list)  # score at each step
    extra:          dict = field(default_factory=dict)   # algorithm-specific info

    def __str__(self):
        return (f"[{self.algorithm:<28}] "
                f"score={self.best_score:.4f} | "
                f"steps={self.steps:5d} | "
                f"restarts={self.restarts:3d} | "
                f"init={self.initial_score:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm 1 — Hill Climbing (Steepest Ascent)
# ══════════════════════════════════════════════════════════════════════════════

def hill_climbing(graph,
                  k: int = DEFAULT_K,
                  allow_sideways: bool = False,
                  max_sideways: int = 5,
                  initial_state: NetworkState | None = None,
                  seed: int | None = None) -> OptimResult:
    """
    Hill Climbing — Steepest Ascent variant.

    At each step: evaluate ALL neighbours, move to the best one.
    If allow_sideways=True: accept sideways moves (equal score) up to
    max_sideways consecutive times. This helps escape flat plateaus.

    Terminates when:
      - No neighbour improves (or ties with allow_sideways) the current score.
      - OR max_sideways consecutive sideways moves are exhausted.

    Parameters
    ----------
    graph           : KarachiGraph instance
    k               : number of active stops
    allow_sideways  : whether to accept equal-score moves
    max_sideways    : max consecutive sideways moves before stopping
    initial_state   : starting state (random if None)
    seed            : RNG seed for reproducibility
    """
    rng = random.Random(seed)
    all_nodes = graph.nodes()
    full_adj  = dict(graph.adjacency)

    state = initial_state or random_state(all_nodes, full_adj, k, rng)
    score = objective(state, all_nodes, full_adj)
    init_score = score

    history   = [score]
    steps     = 0
    sideways  = 0

    while True:
        neighbours = get_neighbours(state, all_nodes, full_adj, k)
        if not neighbours:
            break

        best_nb    = max(neighbours, key=lambda s: objective(s, all_nodes, full_adj))
        best_score = objective(best_nb, all_nodes, full_adj)

        if best_score > score:
            state    = best_nb
            score    = best_score
            sideways = 0
            steps   += 1
            history.append(score)

        elif allow_sideways and best_score == score and sideways < max_sideways:
            state    = best_nb
            sideways += 1
            steps   += 1
            history.append(score)

        else:
            break   # local optimum reached

    return OptimResult(
        algorithm     = f"HC({'sideways' if allow_sideways else 'no-sideways'})",
        best_state    = state,
        best_score    = score,
        initial_score = init_score,
        steps         = steps,
        score_history = history,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm 2 — Random-Restart Hill Climbing
# ══════════════════════════════════════════════════════════════════════════════

def random_restart_hc(graph,
                      k: int = DEFAULT_K,
                      restarts: int = 50,
                      allow_sideways: bool = True,
                      max_sideways: int = 5,
                      seed: int | None = None) -> OptimResult:
    """
    Random-Restart Hill Climbing.

    Runs Hill Climbing from `restarts` independent random starting states.
    Returns the best result found across all runs.

    This partially mitigates local optima: with enough restarts, the
    probability of at least one run landing in the global optimum's
    basin of attraction approaches 1.

    Parameters
    ----------
    graph          : KarachiGraph instance
    k              : number of active stops
    restarts       : number of independent HC runs
    allow_sideways : passed to each HC call
    max_sideways   : passed to each HC call
    seed           : base seed (each restart uses seed+i for reproducibility)
    """
    base_seed  = seed if seed is not None else 42
    best_result = None
    total_steps = 0
    all_scores  = []

    for i in range(restarts):
        result = hill_climbing(
            graph,
            k=k,
            allow_sideways=allow_sideways,
            max_sideways=max_sideways,
            seed=base_seed + i
        )
        total_steps += result.steps
        all_scores.append(result.best_score)

        if best_result is None or result.best_score > best_result.best_score:
            best_result = result

    return OptimResult(
        algorithm     = f"RRHC({restarts} restarts)",
        best_state    = best_result.best_state,
        best_score    = best_result.best_score,
        initial_score = best_result.initial_score,
        steps         = total_steps,
        restarts      = restarts,
        score_history = best_result.score_history,
        extra         = {
            "all_scores":    all_scores,
            "success_rate":  sum(1 for s in all_scores if s >= 0.80) / restarts,
            "avg_score":     sum(all_scores) / len(all_scores),
            "best_score":    max(all_scores),
            "worst_score":   min(all_scores),
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm 3 — Simulated Annealing
# ══════════════════════════════════════════════════════════════════════════════

def simulated_annealing(graph,
                        k: int = DEFAULT_K,
                        T_init: float = 1.0,
                        cooling_rate: float = 0.95,
                        T_min: float = 1e-4,
                        steps_per_temp: int = 20,
                        initial_state: NetworkState | None = None,
                        seed: int | None = None) -> OptimResult:
    """
    Simulated Annealing.

    Accepts worse solutions with probability exp(ΔE / T), where T decreases
    according to a geometric cooling schedule: T ← T × cooling_rate.

    At high T: almost any move is accepted → broad exploration.
    As T → 0: only improvements are accepted → converges to hill climbing.

    Cooling schedule: geometric (T_new = T × r).
      r = 0.90 → fast cooling (more exploitation, fewer total steps)
      r = 0.95 → medium (default, good balance)
      r = 0.99 → slow cooling (more exploration, many steps)

    The number of steps scales as: steps ≈ steps_per_temp × log(T_min/T_init) / log(r)

    Parameters
    ----------
    graph            : KarachiGraph instance
    k                : number of active stops
    T_init           : initial temperature
    cooling_rate     : r in T ← T × r  (0 < r < 1)
    T_min            : stop when T drops below this
    steps_per_temp   : number of random-neighbour evaluations per temperature level
    initial_state    : starting state (random if None)
    seed             : RNG seed
    """
    rng = random.Random(seed)
    all_nodes = graph.nodes()
    full_adj  = dict(graph.adjacency)

    state = initial_state or random_state(all_nodes, full_adj, k, rng)
    score = objective(state, all_nodes, full_adj)
    init_score = score

    best_state = state.copy()
    best_score = score

    history    = [score]
    T          = T_init
    steps      = 0
    accepted   = 0
    rejected   = 0

    while T > T_min:
        for _ in range(steps_per_temp):
            nb       = random_neighbour(state, all_nodes, full_adj, k )
            nb_score = objective(nb, all_nodes, full_adj)
            delta    = nb_score - score

            if delta > 0:
                # Always accept improvements
                state  = nb
                score  = nb_score
                accepted += 1
            else:
                # Accept worse solution with probability exp(delta / T)
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    state  = nb
                    score  = nb_score
                    accepted += 1
                else:
                    rejected += 1

            steps += 1
            history.append(score)

            if score > best_score:
                best_state = state.copy()
                best_score = score

        T *= cooling_rate
        if steps % 1000 == 0:
            print(f"SA progress: {steps} steps", flush=True)
    return OptimResult(
        algorithm     = f"SA(r={cooling_rate})",
        best_state    = best_state,
        best_score    = best_score,
        initial_score = init_score,
        steps         = steps,
        score_history = history,
        extra         = {
            "T_init":       T_init,
            "cooling_rate": cooling_rate,
            "T_min":        T_min,
            "accepted":     accepted,
            "rejected":     rejected,
            "accept_rate":  accepted / steps if steps else 0,
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: run SA with named cooling schedule
# ══════════════════════════════════════════════════════════════════════════════

def sa_with_schedule(graph, k: int = DEFAULT_K,
                     cooling_rate: float = 0.95,
                     seed: int | None = None) -> OptimResult:
    """Thin wrapper: run SA with a specific cooling rate, sensible defaults."""
    return simulated_annealing(
        graph,
        k=k,
        T_init=1.0,
        cooling_rate=cooling_rate,
        T_min=1e-4,
        steps_per_temp=5,
        seed=seed
    )


# ══════════════════════════════════════════════════════════════════════════════
# Score breakdown helper (for reporting best solution)
# ══════════════════════════════════════════════════════════════════════════════

def score_breakdown(state: NetworkState, graph) -> dict:
    """Return the individual component scores for a given state."""
    all_nodes = graph.nodes()
    full_adj  = dict(graph.adjacency)
    cov  = coverage(state, all_nodes, full_adj)
    conn = connectivity(state)
    eff  = efficiency(state)
    total = ALPHA * cov + BETA * conn + GAMMA * eff
    return {
        "coverage":     round(cov,   4),
        "connectivity": round(conn,  4),
        "efficiency":   round(eff,   4),
        "total":        round(total, 4),
        "alpha":        ALPHA,
        "beta":         BETA,
        "gamma":        GAMMA,
    }
