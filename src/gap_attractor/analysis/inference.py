"""Cluster bootstrap and sign test. Clusters are runs, or worlds when agents shared one (they are not independent)."""
from __future__ import annotations

import warnings
from math import comb
from typing import Callable

import numpy as np


def cluster_bootstrap(sums: dict[str, np.ndarray], stat: Callable[[dict], dict], n_boot: int, seed: int,
                      level: float = 0.95) -> dict[str, np.ndarray]:
    """Resample clusters with replacement. `sums` holds additive per-cluster arrays (first axis = cluster);
    `stat` maps totals (each with a leading resample axis) to named statistics. Returns name -> (2, ...) bounds."""
    n_clusters = next(iter(sums.values())).shape[0]
    rng = np.random.default_rng(seed)
    weights = np.zeros((n_boot, n_clusters))
    np.add.at(weights, (np.repeat(np.arange(n_boot), n_clusters), rng.integers(0, n_clusters, n_boot * n_clusters)), 1)
    totals = {k: np.tensordot(weights, v, axes=1) for k, v in sums.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = stat(totals)
    a = (1 - level) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # cells empty in some resamples
        return {k: np.nanquantile(v, [a, 1 - a], axis=0) for k, v in draws.items()}


def sign_test(values: np.ndarray) -> dict:
    """Exact two-sided sign test on per-cluster values (zeros dropped)."""
    pos, neg = int((values > 0).sum()), int((values < 0).sum())
    n = pos + neg
    if n == 0:
        return {"n_pos": 0, "n_neg": 0, "p": 1.0}
    k = min(pos, neg)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"n_pos": pos, "n_neg": neg, "p": p}
