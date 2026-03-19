# blockchain-p2p-privacy-sim

Simulation code accompanying the paper:

> **Privacy Bottleneck in Blockchain P2P Networks: Information-Theoretic Limits of Traffic De-anonymization and Traffic Masking Mechanisms**

This repository contains two experiments that validate the paper's theoretical results under a **Global Passive Observer (GPO)** attacker model on an Erdős-Rényi random graph calibrated to Bitcoin's reachable node count (~12,000 nodes, mean degree 10).

---

## Repository structure

```
blockchain-p2p-privacy-sim/
├── network.py          # Graph construction (Erdős-Rényi) and stats
├── exp1_compare.py     # Exp 1 — four-strategy source accuracy + M_A tracking
├── exp2_paradox.py     # Exp 2 — individual adoption paradox
├── plot_all.py         # Figure generation (reads results/, writes figures/)
├── run_all.py          # Entry point: runs Exp 1 → Exp 2 → figures
├── requirements.txt    # Python dependencies
├── results/            # JSON output (auto-created on first run)
└── figures/            # PDF + PNG figures (auto-created on first run)
```

---

## Quick start

```bash
pip install -r requirements.txt
python run_all.py
```

Figures are written to `figures/` and raw results to `results/`.

---

## Experiments

### Experiment 1 — Four-strategy source identification accuracy

**File:** `exp1_compare.py`
**Network:** n = 12,000 nodes, Erdős-Rényi G(n, p) with p = 10/11,999 (mean degree ≈ 10)
**Trials:** 300 per strategy

All four strategies share the same GPO observation model:

| Strategy | Description |
|---|---|
| No Defense | Standard gossip; source broadcasts at t = 0 |
| Dandelion++ | Source sends to one random stem neighbour at t = 0 |
| M_G | All nodes generate Poisson(λ_cover = 1 msg/s/edge) cover traffic |
| M_A | All nodes track λ̄(t) via a window estimator and set λ_A(t) = λ̄(t) + δ |

**Key results:**

| Strategy | Accuracy | 95% CI (Clopper-Pearson) |
|---|---|---|
| No Defense | 100% | [98.78%, 100.00%] |
| Dandelion++ | 100% | [98.78%, 100.00%] |
| M_G | ≈0.3% | [0.01%, 1.84%] |
| M_A | 0.0% | [0.00%, 1.22%] |

Both M_G and M_A reduce accuracy to the random-guess baseline (1/n ≈ 0.008%), consistent with the paper's I(O; Y) = 0 theorem. Dandelion++ achieves no privacy against a GPO because the source is identified at the first packet, before any stem hops occur.

The experiment also measures M_A's adaptive tracking: mean tracking error < 0.01 msg/s/edge across 60 simulated time windows.

### Experiment 2 — Individual adoption paradox

**File:** `exp2_paradox.py`
**Network:** n = 1,000 nodes
**Trials:** 500 per adoption rate α

Validates Theorem 3: an adopter's effective anonymity set is

```
E[AS(α)] = α² n + 1 − α
```

which is proportional to α²n, not n. At α = 5%, AS ≈ 50 nodes (5% of n), not 1,000. The simulation uses a variance classifier (threshold τ = 1 s²) applied to real inter-packet interval (IPI) sequences:

- Adopters (M_G): deterministic period 1/λ_cover = 1 s, IPI variance = 0 s²
- Non-adopters: Poisson(λ_real = 0.1 msg/s), IPI variance ≈ 100 s²

The two-order-of-magnitude variance gap gives classifier accuracy > 99% at all α values, confirming that the simulation reflects the masking mechanism itself rather than classifier noise.

---

## Key parameters

| Parameter | Value | Description |
|---|---|---|
| N_NODES (Exp 1) | 12,000 | Bitcoin reachable node count |
| N_NODES (Exp 2) | 1,000 | Anonymity set analysis scale |
| ER_P | 10 / (n−1) | Edge probability, mean degree ≈ 10 |
| λ_cover | 1.0 msg/s/edge | M_G cover traffic rate |
| λ_real | 0.1 msg/s/edge | Natural message arrival rate |
| δ | 0.01 msg/s/edge | M_A safety margin |
| prop. delay | Uniform[50, 500] ms | Per-edge propagation delay |
| SEED | 42 | Global random seed |

---

## Output figures

| File | Description | In paper |
|---|---|---|
| `fig_exp1_accuracy.pdf` | Four-strategy accuracy bar chart with 95% CIs | Yes |
| `fig_exp2_paradox.pdf` | Normalised anonymity set vs adoption rate | Yes |
| `fig_exp2_clf_accuracy_diag.pdf` | Variance classifier accuracy diagnostic | No |

---

## Dependencies

```
networkx >= 2.8
numpy    >= 1.23
scipy    >= 1.9
matplotlib >= 3.6
```

Install via:

```bash
pip install -r requirements.txt
```

Python 3.9+ recommended.

---

## Reproducibility

All random seeds are fixed (global `SEED = 42`). Running `python run_all.py` from this directory produces results and figures that are bit-for-bit identical to those reported in the paper.

---

## License

MIT License. See `LICENSE` for details.
