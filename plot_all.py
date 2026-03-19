"""
plot_all.py — Paper figure generation

Reads only JSON files from the results/ directory; contains no
experiment logic. Separation of concerns: computation (exp1/exp2)
and plotting are strictly decoupled.

Output figures
--------------
  fig_exp1_accuracy           — Exp 1 four-strategy source accuracy (in paper)
  fig_exp2_paradox            — Exp 2 individual adoption paradox theory vs sim (in paper)
  fig_exp2_clf_accuracy_diag  — Exp 2 classifier accuracy diagnostic (not in paper)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path(__file__).parent / "results"
figures_dir = Path(__file__).parent / "figures"
figures_dir.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

STRATEGY_LABELS = {
    "no_defense": "No Defense",
    "dandelion":  "Dandelion++",
    "mg":         r"$\mathcal{M}_G$",
    "ma":         r"$\mathcal{M}_A$",
}
COLORS = {
    "no_defense": "#d62728",
    "dandelion":  "#ff7f0e",
    "mg":         "#1f77b4",
    "ma":         "#2ca02c",
}


# ──────────────────────────────────────────────
# Fig 1: Exp 1 source identification accuracy
# ──────────────────────────────────────────────

def plot_exp1_accuracy(results):
    """
    Parameters
    ----------
    results : list of dicts, each with keys:
        strategy, accuracy, accuracy_ci_low, accuracy_ci_high,
        n_correct, n_nodes, n_trials
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    strategies = [r["strategy"] for r in results]
    labels     = [STRATEGY_LABELS[s] for s in strategies]
    accs       = [r["accuracy"] * 100 for r in results]
    ci_lows    = [r["accuracy_ci_low"]  * 100 for r in results]
    ci_highs   = [r["accuracy_ci_high"] * 100 for r in results]
    colors     = [COLORS[s] for s in strategies]
    n          = results[0]["n_nodes"]

    x = np.arange(len(strategies))
    yerr_low  = np.clip([a - l for a, l in zip(accs, ci_lows)],  0, None)
    yerr_high = np.clip([h - a for a, h in zip(accs, ci_highs)], 0, None)

    bars = ax.bar(x, accs, color=colors, width=0.55, alpha=0.85,
                  yerr=[yerr_low, yerr_high], capsize=5,
                  error_kw={"elinewidth": 1.5, "ecolor": "black"})

    ax.axhline(100 / n, color="gray", linestyle="--", linewidth=1.2,
               label=f"Random guess (1/n = {100/n:.3f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Source Identification Accuracy (%)")
    ax.set_xlabel("Defense Strategy")
    ax.set_title("Source Traceability Under Global Passive Observer\n"
                 f"(n={n:,}, {results[0]['n_trials']} trials, 95% Clopper-Pearson CI)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 115)

    for bar, acc, n_correct in zip(bars, accs, [r["n_correct"] for r in results]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{acc:.1f}%\n({n_correct}/{results[0]['n_trials']})",
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"fig_exp1_accuracy.{ext}")
    print("Saved: fig_exp1_accuracy.pdf / .png")
    plt.close(fig)


# ──────────────────────────────────────────────
# Fig 2: Exp 2 individual adoption paradox
# ──────────────────────────────────────────────

def plot_exp2_paradox(results):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    alphas       = [r["alpha"] for r in results]
    sim_norm     = np.array([r["mean_as_norm"] for r in results])
    ci_low_norm  = np.clip([r["ci_low_norm"]  for r in results], 0, None)
    ci_high_norm = np.array([r["ci_high_norm"] for r in results])
    n = results[0]["n_nodes"]

    # Theoretical curve E[AS(alpha)]/n = alpha^2 + (1-alpha)/n
    alpha_fine  = np.linspace(0, 1, 300)
    theory_fine = alpha_fine**2 + (1 - alpha_fine) / n

    ax.plot(alpha_fine, theory_fine, "k-", linewidth=2,
            label=r"Theory: $E[AS(\alpha)]/n = \alpha^2 + (1-\alpha)/n$")

    yerr_low  = np.clip(sim_norm - ci_low_norm,  0, None)
    yerr_high = np.clip(ci_high_norm - sim_norm, 0, None)
    ax.errorbar(alphas, sim_norm,
                yerr=[yerr_low, yerr_high],
                fmt="o", color="#1f77b4", markersize=5, capsize=3,
                linewidth=1, elinewidth=1.2,
                label="Simulation (variance classifier, 95% CI)")

    # Annotate alpha = 0.05
    a05 = 0.05
    y05 = a05**2 + (1 - a05) / n
    ax.annotate(f"alpha=5%: AS/n~{y05:.4f}\n(AS~{y05*n:.0f}/{n})",
                xy=(a05, y05), xytext=(0.15, 0.12),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9, color="gray")

    ax.set_xlabel(r"Adoption Rate $\alpha$")
    ax.set_ylabel(r"Normalized Anonymity Set $E[AS(\alpha)]/n$")
    ax.set_title("Individual Adoption Paradox\n"
                 r"Adopter's AS $= \alpha n \ll n$ for small $\alpha$")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.02, 1.05)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"fig_exp2_paradox.{ext}")
    print("Saved: fig_exp2_paradox.pdf / .png")
    plt.close(fig)


# ──────────────────────────────────────────────
# Diagnostic: variance classifier accuracy (not in paper)
# ──────────────────────────────────────────────

def plot_exp2_clf_accuracy(results):
    if "mean_classifier_accuracy" not in results[0]:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    alphas = [r["alpha"] for r in results]
    accs   = [r["mean_classifier_accuracy"] * 100 for r in results]
    ax.plot(alphas, accs, "o-", color="#2ca02c", linewidth=1.5, markersize=5)
    ax.set_xlabel(r"Adoption Rate $\alpha$")
    ax.set_ylabel("Variance Classifier Accuracy (%)")
    ax.set_title("Diagnostic: Variance Classifier Accuracy vs alpha\n(not in paper)")
    ax.set_ylim(90, 101)
    ax.set_xlim(0, 1.02)
    ax.axhline(100, color="gray", linestyle="--", linewidth=1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"fig_exp2_clf_accuracy_diag.{ext}")
    print("Saved: fig_exp2_clf_accuracy_diag.pdf / .png (diagnostic, not in paper)")
    plt.close(fig)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    print("=" * 50)
    print("Generating paper figures")
    print("=" * 50)

    exp1_path = results_dir / "exp1_results.json"
    if exp1_path.exists():
        with open(exp1_path) as f:
            exp1_raw = json.load(f)
        # exp1_results.json uses a nested format; convert to list for plotting
        sr      = exp1_raw["strategy_results"]
        n_nodes = exp1_raw["graph_stats"]["n_nodes"]
        n_tri   = exp1_raw["parameters"]["n_trials"]
        exp1_list = []
        for strat in ["no_defense", "dandelion", "mg", "ma"]:
            r = sr[strat]
            exp1_list.append({
                "strategy":         strat,
                "accuracy":         r["accuracy"],
                "accuracy_ci_low":  r["ci_lower"],
                "accuracy_ci_high": r["ci_upper"],
                "n_correct":        r["correct"],
                "n_nodes":          n_nodes,
                "n_trials":         n_tri,
            })
        plot_exp1_accuracy(exp1_list)
    else:
        print(f"Warning: {exp1_path} not found, skipping Exp 1")

    exp2_path = results_dir / "exp2_results.json"
    if exp2_path.exists():
        with open(exp2_path) as f:
            exp2 = json.load(f)
        plot_exp2_paradox(exp2)
        plot_exp2_clf_accuracy(exp2)
    else:
        print(f"Warning: {exp2_path} not found, skipping Exp 2")

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
