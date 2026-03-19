"""
exp2_paradox.py — Individual adoption paradox

Validates Theorem 3 (individual paradox) using real inter-packet interval
(IPI) sequences combined with a variance-based classifier.

Experimental design
-------------------
- Each node generates an IPI sequence over a T_obs-second window
  according to its adoption status:
    Adopter (M_G):   deterministic period 1/lambda_cover = 1 s -> IPI variance ~ 0
    Non-adopter:     Poisson(lambda_real = 0.1 msg/s)  -> IPI variance ~ 100 s^2
- The attacker runs a variance classifier (IPI variance < tau -> adopter)
- The effective anonymity set size AS(alpha) is computed from
  the classification outcome for the chosen source node

Theoretical prediction (Theorem 3 corollary)
---------------------------------------------
  E[AS(alpha)] = alpha^2 * n + 1 - alpha

Non-adopter source anonymity set equals 1 by the independent result of
Exp 1 (no cover traffic -> GPO achieves 100% source identification accuracy).
"""

import numpy as np
import random
import json
from pathlib import Path
from network import build_er_graph

# ──────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────
N_NODES            = 1000
ER_P               = 10 / 999
T_OBS              = 50.0          # observation window (s)
LAMBDA_COVER       = 1.0           # adopter sending rate (msg/s); IPI = 1 s, var ~ 0
LAMBDA_REAL        = 0.1           # non-adopter natural rate (msg/s); IPI var ~ 100 s^2
VARIANCE_THRESHOLD = 1.0           # classification threshold (s^2)
N_ALPHA_POINTS     = 20            # number of alpha scan points (uniform on [0.02, 1.0])
N_TRIALS           = 500           # repetitions per alpha value
SEED               = 42

results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# IPI sequence generation
# ──────────────────────────────────────────────

def generate_ipi_adopter():
    """
    Adopter under M_G: deterministic period 1/lambda_cover.
    Returns an IPI sequence of length floor(T_obs * lambda_cover) - 1,
    with exactly zero variance.
    """
    n_packets = int(T_OBS * LAMBDA_COVER)   # = 50
    if n_packets < 2:
        return np.array([1.0 / LAMBDA_COVER])
    return np.full(n_packets - 1, 1.0 / LAMBDA_COVER)


def generate_ipi_non_adopter(rng):
    """
    Non-adopter: Poisson(lambda_real) arrivals, IPI ~ Exp(1/lambda_real).
    Returns the IPI sequence within the T_obs window;
    expected packet count = T_obs * lambda_real = 5.
    """
    intervals = []
    t = 0.0
    while True:
        dt = rng.expovariate(LAMBDA_REAL)
        t += dt
        if t >= T_OBS:
            break
        intervals.append(dt)
    return np.array(intervals) if intervals else np.array([])


# ──────────────────────────────────────────────
# Variance classifier
# ──────────────────────────────────────────────

def classify_node(ipi_seq, threshold=VARIANCE_THRESHOLD):
    """
    Variance classifier: IPI variance < threshold -> adopter (True);
    otherwise -> non-adopter (False).
    If the IPI sequence has fewer than 2 samples, conservatively
    classify as non-adopter.
    """
    if len(ipi_seq) < 2:
        return False
    return float(np.var(ipi_seq)) < threshold


# ──────────────────────────────────────────────
# Single trial
# ──────────────────────────────────────────────

def run_trial(alpha, n, rng):
    """
    Single trial:
    1. Assign adoption status independently to each of n nodes (Bernoulli(alpha))
    2. Generate the true IPI sequence for each node and run the variance classifier
    3. Pick a random source node and compute its effective anonymity set size

    Returns: (as_size, clf_accuracy, source_is_adopter, n_classified_adopters)
    """
    is_adopter = np.random.random(n) < alpha
    source_idx = rng.randint(0, n - 1)

    classified_adopter = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_adopter[i]:
            ipi = generate_ipi_adopter()
        else:
            ipi = generate_ipi_non_adopter(rng)
        classified_adopter[i] = classify_node(ipi)

    n_cls_adopters     = int(np.sum(classified_adopter))
    source_cls_adopter = bool(classified_adopter[source_idx])

    if source_cls_adopter:
        # Source classified as adopter: anonymity set = all nodes
        # classified as adopters
        as_size = float(max(1, n_cls_adopters))
    else:
        # Source classified as non-adopter: GPO can pinpoint via timestamps
        # (Exp 1 confirms 100% accuracy with no cover traffic)
        as_size = 1.0

    clf_acc = float(np.sum(classified_adopter == is_adopter)) / n
    return as_size, clf_acc, bool(is_adopter[source_idx]), n_cls_adopters


# ──────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────

def run_exp2():
    print("=" * 60)
    print("Exp 2: Individual adoption paradox (variance classifier + real IPI)")
    print("=" * 60)
    print(f"Adopter IPI variance = 0 s^2 (deterministic period {1/LAMBDA_COVER:.1f} s)")
    print(f"Non-adopter IPI variance ~ {1/LAMBDA_REAL**2:.0f} s^2 "
          f"(Poisson, mean interval {1/LAMBDA_REAL:.1f} s)")
    print(f"Classification threshold tau = {VARIANCE_THRESHOLD} s^2\n")

    rng = random.Random(SEED)
    np.random.seed(SEED)

    G = build_er_graph(N_NODES, ER_P, SEED)
    n = G.number_of_nodes()

    alpha_values = np.linspace(0.02, 1.0, N_ALPHA_POINTS)
    all_results  = []

    for alpha in alpha_values:
        as_values, clf_accs = [], []
        for _ in range(N_TRIALS):
            as_val, clf_acc, _, _ = run_trial(alpha, n, rng)
            as_values.append(as_val)
            clf_accs.append(clf_acc)

        as_arr  = np.array(as_values)
        mean_as = float(np.mean(as_arr))

        # Bootstrap 95% CI for the mean estimator
        boot_rng   = np.random.default_rng(SEED + int(round(alpha * 1000)))
        boot_means = [
            float(np.mean(boot_rng.choice(as_arr, size=len(as_arr), replace=True)))
            for _ in range(2000)
        ]
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))

        theory_as = alpha**2 * n + 1 - alpha

        result = {
            "alpha":                    float(alpha),
            "mean_as":                  mean_as,
            "mean_as_norm":             mean_as / n,
            "std_as":                   float(np.std(as_arr)),
            "ci_low":                   ci_lo,
            "ci_low_norm":              ci_lo / n,
            "ci_high":                  ci_hi,
            "ci_high_norm":             ci_hi / n,
            "theory_as":                float(theory_as),
            "theory_as_norm":           float(theory_as / n),
            "mean_classifier_accuracy": float(np.mean(clf_accs)),
            "n_trials":                 N_TRIALS,
            "n_nodes":                  int(n),
            "params": {
                "t_obs":              T_OBS,
                "lambda_cover":       LAMBDA_COVER,
                "lambda_real":        LAMBDA_REAL,
                "variance_threshold": VARIANCE_THRESHOLD,
            },
        }
        all_results.append(result)
        print(f"  alpha={alpha:.3f}: sim E[AS]={mean_as:6.1f}  theory={theory_as:6.1f}  "
              f"clf_acc={result['mean_classifier_accuracy']*100:.1f}%")

    out = results_dir / "exp2_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")
    return all_results


if __name__ == "__main__":
    run_exp2()
