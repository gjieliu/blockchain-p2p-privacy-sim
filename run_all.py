"""
run_all.py — Main entry point

Runs in sequence: Exp 1 (source accuracy comparison) ->
Exp 2 (individual adoption paradox) -> figure generation.

Usage (from the en/ directory):
    python run_all.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from exp1_compare import run_exp1
from exp2_paradox import run_exp2
from plot_all import main as plot_main


def main():
    print("\n" + "=" * 60)
    print("Blockchain P2P Layer Privacy Simulation (Global Passive Observer)")
    print("=" * 60 + "\n")

    print(">>> Exp 1: Four-strategy source identification accuracy")
    run_exp1()

    print("\n>>> Exp 2: Individual adoption paradox")
    run_exp2()

    print("\n>>> Generating all figures")
    plot_main()

    print("\n" + "=" * 60)
    print("All experiments complete. Results in results/, figures in figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
