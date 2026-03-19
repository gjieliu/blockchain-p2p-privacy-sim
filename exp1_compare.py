"""
exp1_compare.py — Four-strategy source identification accuracy comparison

Attacker model: Global Passive Observer (GPO)
Network scale:  n = 12,000 (realistic Bitcoin reachable node count)

Unified simulation framework
-----------------------------
All four strategies are executed through the same function
simulate_and_attack_vec(), using an identical GPO timestamp
observation model:

  T_{v,u}^obs = wait_{v,u} + eps_{v,u}     eps ~ Uniform(50 ms, 500 ms)
  T_v^obs     = min_{u in N(v)} T_{v,u}^obs
  predicted   = argmin_v T_v^obs

The four strategies differ only in the distribution of wait_{v,u}:

  no_defense : source broadcasts on all outgoing edges at t=0 (wait=0);
               all other nodes/edges have wait=inf
  dandelion  : source sends to a single random stem neighbour at t=0;
               all other nodes/edges have wait=inf
  M_G        : all edges on all nodes draw wait ~ Exp(lambda_cover)
  M_A        : all edges draw wait ~ Exp(lambda_A(t)), where lambda_A is
               estimated from a window-based Poisson arrival count

Implementation
--------------
Uses numpy vectorisation (np.minimum.reduceat) to compute per-node minimum
observation times in batch, avoiding per-node Python loops.
With n=12,000 and mean degree 10, there are ~120,000 directed edges;
each trial takes ~2 ms, and 300 trials x 4 strategies complete in ~10 s.
"""

import numpy as np
import json
from pathlib import Path
from scipy.stats import beta as scipy_beta
from network import build_er_graph, graph_stats

# ──────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────
N_NODES      = 12000          # Bitcoin reachable nodes: ~12,000–15,000
ER_P         = 10 / 11999     # Edge prob for mean degree ≈ 10
LAMBDA_REAL  = 0.1            # Natural message arrival rate (msg/s)
LAMBDA_COVER = 1.0            # M_G cover traffic rate (msg/s/edge)
DELTA        = 0.01           # M_A extra margin (msg/s/edge)
LAMBDA_A     = LAMBDA_REAL + DELTA   # M_A adaptive rate = 0.11 msg/s/edge
Q_DANDELION  = 0.9            # Dandelion++ stem forwarding probability
                              # (protocol parameter, recorded for reference;
                              #  not used in GPO simulation: the GPO identifies
                              #  the source at the first packet, before any
                              #  subsequent stem hops occur)
N_TRIALS     = 300
SEED         = 42

PROP_DELAY_MIN = 0.05         # Propagation delay lower bound: 50 ms
PROP_DELAY_MAX = 0.50         # Propagation delay upper bound: 500 ms

results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# Graph structure precomputation
# ──────────────────────────────────────────────

def precompute_graph_structure(G):
    """
    Convert graph G to the directed-edge array format required for
    vectorised simulation.

    Returns
    -------
    dict with keys:
        nodes          ndarray(n,) int32   node labels (0..n-1)
        n              int                 number of nodes
        src, dst       ndarray(n_dir,)     directed edges src->dst, sorted by src
        group_starts   ndarray(n,)         start index in src array for each node
        node_has_edges ndarray(n,) bool    whether each node has outgoing edges
        n_directed     int                 total directed edges (= 2 * undirected edges)
    """
    nodes_list = sorted(G.nodes())
    n = len(nodes_list)

    eu, ev = [], []
    for u, v in G.edges():
        eu.append(u); ev.append(v)   # undirected edge -> two directed edges
        eu.append(v); ev.append(u)

    n_directed = len(eu)
    src_arr = np.array(eu, dtype=np.int32)
    dst_arr = np.array(ev, dtype=np.int32)

    order   = np.argsort(src_arr, kind='stable')
    src_arr = src_arr[order]
    dst_arr = dst_arr[order]

    # group_starts[i]: index of first occurrence of node i in src_arr
    group_starts = np.searchsorted(src_arr, np.arange(n)).astype(np.int32)

    node_has_edges = np.zeros(n, dtype=bool)
    if n_directed > 0:
        node_has_edges[src_arr] = True

    return {
        'nodes':          np.array(nodes_list, dtype=np.int32),
        'n':              n,
        'src':            src_arr,
        'dst':            dst_arr,
        'group_starts':   group_starts,
        'node_has_edges': node_has_edges,
        'n_directed':     n_directed,
    }


# ──────────────────────────────────────────────
# M_A time-varying background traffic schedule
# ──────────────────────────────────────────────
MA_LAMBDA_LOW  = 0.05    # msg/s/edge (low-load phase)
MA_LAMBDA_HIGH = 0.20    # msg/s/edge (high-load phase)
MA_WINDOW_SIZE = 10.0    # seconds per window
MA_SCHEDULE = [MA_LAMBDA_LOW] * 20 + [MA_LAMBDA_HIGH] * 20 + [MA_LAMBDA_LOW] * 20


# ──────────────────────────────────────────────
# Vectorised unified simulation function
# ──────────────────────────────────────────────

def simulate_and_attack_vec(gs, source_idx, strategy, lambda_cover, np_rng, trial_idx=None):
    """
    Vectorised GPO simulation. All four strategies share the same
    observation model; only the wait distribution differs.

    T_{v,u}^obs = wait_{v,u} + eps_{v,u}
    predicted   = argmin_v min_{u in N(v)} T_{v,u}^obs

    Implementation: per-edge obs vector arithmetic +
    np.minimum.reduceat for batch per-node minimisation.

    Returns: predicted (all strategies), or (predicted, lambda_A)
             (M_A only).
    """
    n            = gs['n']
    group_starts = gs['group_starts']
    node_has     = gs['node_has_edges']
    n_dir        = gs['n_directed']

    # Sample propagation delay eps_{v,u} independently for each directed edge
    delays = np_rng.uniform(PROP_DELAY_MIN, PROP_DELAY_MAX, size=n_dir)

    lambda_a_used = None

    if strategy == "no_defense":
        # Source broadcasts on all outgoing edges at t=0;
        # all other nodes have wait=inf (have not yet received the packet)
        obs = np.full(n_dir, np.inf)
        s0  = int(group_starts[source_idx])
        s1  = int(group_starts[source_idx + 1]) if source_idx + 1 < n else n_dir
        obs[s0:s1] = delays[s0:s1]

    elif strategy == "dandelion":
        # Dandelion++ under GPO: source sends the first packet to a single
        # random stem neighbour at t=0 (wait=0); all other nodes/edges have
        # wait=inf. Q_DANDELION=0.9 governs subsequent stem hops, but the
        # GPO identifies the source at the first packet, before those occur.
        obs = np.full(n_dir, np.inf)
        s0  = int(group_starts[source_idx])
        s1  = int(group_starts[source_idx + 1]) if source_idx + 1 < n else n_dir
        if s1 > s0:
            off = int(np_rng.integers(0, s1 - s0))
            obs[s0 + off] = delays[s0 + off]   # only the chosen stem direction

    elif strategy == "mg":
        # All nodes continuously generate Poisson(lambda_cover) cover traffic
        waits = np_rng.exponential(1.0 / lambda_cover, size=n_dir)
        obs   = waits + delays

    else:  # "ma"
        # M_A: sample a random window from the schedule, estimate lambda_bar,
        # set lambda_A = lambda_bar + delta
        n_win = len(MA_SCHEDULE)
        w_idx = int(np_rng.integers(0, n_win))
        lr    = MA_SCHEDULE[w_idx]
        n_ue  = n_dir // 2   # number of undirected edges for Poisson estimation
        est_rng       = np.random.default_rng(SEED + trial_idx * n_win + w_idx)
        arrivals      = est_rng.poisson(lr * MA_WINDOW_SIZE, size=n_ue)
        lambda_bar_est = float(np.mean(arrivals)) / MA_WINDOW_SIZE
        lambda_a_used  = lambda_bar_est + DELTA
        waits = np_rng.exponential(1.0 / lambda_a_used, size=n_dir)
        obs   = waits + delays

    # T_obs[v] = minimum observed time across all outgoing edges of v.
    # Append a sentinel inf so isolated nodes' group_starts point to a valid slot.
    obs_padded = np.concatenate([obs, [np.inf]])
    gs_cl      = np.minimum(group_starts, n_dir).astype(np.intp)
    T_obs      = np.minimum.reduceat(obs_padded, gs_cl)
    T_obs[~node_has] = np.inf   # fix isolated nodes (extremely rare in ER graph)

    pred_idx  = int(np.argmin(T_obs))
    predicted = int(gs['nodes'][pred_idx])

    if strategy == "ma":
        return predicted, lambda_a_used
    return predicted


# ──────────────────────────────────────────────
# M_A adaptive mechanism verification
# ──────────────────────────────────────────────

def simulate_ma_mechanism_verification(G):
    """
    Verify M_A's adaptivity: can M_A continuously track lambda_bar(t) + delta
    under time-varying background traffic?

    For each window w in MA_SCHEDULE (60 windows total):
      1. Simulate observed traffic via Poisson arrivals, estimate lambda_bar_w.
      2. Compute lambda_A(w) = lambda_bar_w + delta.
      3. Compute tracking error err(w) = |lambda_A(w) - (lambda_real(w) + delta)|.

    Returns
    -------
    dict with keys: window_lambdas, estimated_lambda_a, tracking_errors,
                    mean_tracking_error, max_tracking_error.
    """
    n_dir = sum(1 for _ in G.edges()) * 2   # approximate directed edge count
    n_ue  = n_dir // 2                       # undirected edges for Poisson estimation
    rng   = np.random.default_rng(SEED + 9999)

    n_win           = len(MA_SCHEDULE)
    window_lambdas  = np.array(MA_SCHEDULE, dtype=float)
    estimated_la    = np.zeros(n_win)

    for w_idx in range(n_win):
        lr       = MA_SCHEDULE[w_idx]
        arrivals = rng.poisson(lr * MA_WINDOW_SIZE, size=n_ue)
        estimated_la[w_idx] = float(np.mean(arrivals)) / MA_WINDOW_SIZE + DELTA

    tracking_errors = np.abs(estimated_la - (window_lambdas + DELTA))

    return {
        "window_lambdas":     window_lambdas.tolist(),
        "estimated_lambda_a": estimated_la.tolist(),
        "tracking_errors":    tracking_errors.tolist(),
        "mean_tracking_error": float(np.mean(tracking_errors)),
        "max_tracking_error":  float(np.max(tracking_errors)),
    }


# ──────────────────────────────────────────────
# Clopper-Pearson exact binomial confidence interval
# ──────────────────────────────────────────────

def clopper_pearson_ci(k, n, confidence=0.95):
    """
    Compute the Clopper-Pearson exact confidence interval for a
    binomial proportion.

    Parameters
    ----------
    k          : int    number of successes
    n          : int    total number of trials
    confidence : float  confidence level (default 0.95)

    Returns
    -------
    (lower, upper) — confidence interval endpoints (float)

    Notes
    -----
    Endpoints are quantiles of the Beta distribution:
      lower = Beta(alpha/2;   k,   n-k+1)
      upper = Beta(1-alpha/2; k+1, n-k)
    Edge cases: k=0 gives lower=0; k=n gives upper=1.
    """
    alpha = 1.0 - confidence
    if k == 0:
        lower = 0.0
    else:
        lower = float(scipy_beta.ppf(alpha / 2, k, n - k + 1))
    if k == n:
        upper = 1.0
    else:
        upper = float(scipy_beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper


# ──────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────

def run_exp1():
    """
    Main function for Experiment 1: run four-strategy GPO simulation on
    a Bitcoin-scale ER graph, report source identification accuracy with
    95% Clopper-Pearson CIs, and run M_A adaptive tracking verification.

    Results are written to results/exp1_results.json.
    """
    print("=" * 60)
    print("Exp 1: Four-strategy source identification accuracy (GPO)")
    print(f"n={N_NODES}, mean degree≈10, trials={N_TRIALS}")
    print("=" * 60)

    # ── 1. Generate random graph (fixed seed, shared across all strategies) ──
    print("\n[1/3] Generating Erdos-Renyi graph ...", end=" ", flush=True)
    G = build_er_graph(N_NODES, ER_P, seed=SEED)
    stats = graph_stats(G)
    print(f"done. nodes={stats['n_nodes']}, edges={stats['n_edges']}, "
          f"mean_deg={stats['mean_degree']:.2f}, max_deg={stats['max_degree']}")

    gs    = precompute_graph_structure(G)
    nodes = gs['nodes']

    # ── 2. Run four-strategy simulation ──
    print(f"\n[2/3] Running {N_TRIALS} trials x 4 strategies ...")
    strategies = ["no_defense", "dandelion", "mg", "ma"]
    strategy_labels = {
        "no_defense": "No Defense",
        "dandelion":  "Dandelion++",
        "mg":         "M_G (global cover traffic)",
        "ma":         "M_A (adaptive cover traffic)",
    }

    np_rng  = np.random.default_rng(SEED)
    results = {}

    for strat in strategies:
        correct = 0
        lambda_a_list = []

        source_indices = np_rng.integers(0, gs['n'], size=N_TRIALS)

        for t, src_idx in enumerate(source_indices):
            src_node = int(nodes[src_idx])
            if strat == "ma":
                pred, la = simulate_and_attack_vec(
                    gs, src_idx, strat, LAMBDA_COVER, np_rng, trial_idx=t)
                lambda_a_list.append(la)
            else:
                pred = simulate_and_attack_vec(
                    gs, src_idx, strat, LAMBDA_COVER, np_rng, trial_idx=t)
            if pred == src_node:
                correct += 1

        acc    = correct / N_TRIALS
        lo, hi = clopper_pearson_ci(correct, N_TRIALS)
        results[strat] = {
            "correct":  correct,
            "n_trials": N_TRIALS,
            "accuracy": acc,
            "ci_lower": lo,
            "ci_upper": hi,
        }
        if strat == "ma":
            results[strat]["lambda_a_mean"] = float(np.mean(lambda_a_list))
            results[strat]["lambda_a_std"]  = float(np.std(lambda_a_list))

        print(f"  {strategy_labels[strat]:35s}  acc={acc*100:6.2f}%  "
              f"95%CI=[{lo*100:.2f}%, {hi*100:.2f}%]")

    # ── 3. M_A adaptive tracking verification ──
    print("\n[3/3] M_A adaptive tracking verification ...")
    ma_track = simulate_ma_mechanism_verification(G)
    print(f"  mean tracking error={ma_track['mean_tracking_error']:.5f} msg/s/edge, "
          f"max={ma_track['max_tracking_error']:.5f} msg/s/edge")

    # ── Save results ──
    output = {
        "experiment":  "exp1_four_strategy_comparison",
        "parameters": {
            "n_nodes":        N_NODES,
            "er_p":           ER_P,
            "lambda_cover":   LAMBDA_COVER,
            "lambda_real":    LAMBDA_REAL,
            "delta":          DELTA,
            "lambda_a":       LAMBDA_A,
            "n_trials":       N_TRIALS,
            "seed":           SEED,
            "prop_delay_min": PROP_DELAY_MIN,
            "prop_delay_max": PROP_DELAY_MAX,
        },
        "graph_stats":      stats,
        "strategy_results": results,
        "ma_tracking":      ma_track,
    }

    out_path = results_dir / "exp1_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {out_path}")
    return output


if __name__ == "__main__":
    run_exp1()
