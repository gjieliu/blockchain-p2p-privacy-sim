"""
network.py — Network construction utilities

Provides functions for building synthetic random graphs and
computing basic graph statistics used across all experiments.
"""

import networkx as nx
import numpy as np


def build_er_graph(n: int, p: float, seed: int = 42) -> nx.Graph:
    """
    Build an Erdős-Rényi random graph G(n, p).

    Parameters
    ----------
    n    : number of nodes
    p    : edge probability (each pair connected independently with prob p)
    seed : random seed for reproducibility

    Returns
    -------
    nx.Graph — undirected graph with nodes labelled 0..n-1
    """
    return nx.erdos_renyi_graph(n, p, seed=seed)


def graph_stats(G: nx.Graph) -> dict:
    """
    Compute basic statistics for graph G.

    Returns
    -------
    dict with keys:
        n_nodes     — number of nodes
        n_edges     — number of edges (undirected)
        mean_degree — average node degree
        max_degree  — maximum node degree
    """
    degrees = [d for _, d in G.degree()]
    return {
        "n_nodes":     G.number_of_nodes(),
        "n_edges":     G.number_of_edges(),
        "mean_degree": float(np.mean(degrees)),
        "max_degree":  int(np.max(degrees)) if degrees else 0,
    }
