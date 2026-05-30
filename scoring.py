"""
scoring.py — Measurement harness (SHARED, method-agnostic).

Moritz owns this. It is the single instrument the whole project reports
through, so the metrics are defined here once, unambiguously. It scores ANY
method's recommendations against the SVD baseline, so Tim's Tang output plugs
into the exact same harness (this is the seam between Moritz and Tim).

Metrics (per Project-Draft.tex, Section 6 and Table 1):

  precision@10
      Fraction of a user's top-10 recommendations shared with the full SVD
      baseline, averaged over a sample of users and the six seeds.
          p@10(method, baseline) = |top10_method ∩ top10_baseline| / 10

  subspace overlap
      How well a recovered subspace V_hat captures the true top-k subspace:
          overlap = (1/k) * ||V_k^T V_hat||_F^2   ∈ [0, 1]
      1.0 means the spans coincide. (Both V_k and V_hat must have
      orthonormal columns for the [0,1] normalization to hold.)

Public API:
    precision_at_k(recs_method, recs_baseline, t=10) -> float
    subspace_overlap(V_k, V_hat) -> float
    evaluate_method(method_fn, seeds, n_users, ...) -> dict
        Generic driver: give it a callable that returns (recs, V_hat) for a
        matrix, get back mean/std precision and overlap across seeds.
"""

from __future__ import annotations

import numpy as np

from data import make_matrix, SEEDS, K_GROUPS
from baseline import recommend_all, fit_subspace


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def precision_at_k(recs_method: np.ndarray, recs_baseline: np.ndarray,
                   t: int = 10) -> float:
    """Mean precision@t between two (n_users, >=t) recommendation arrays.

    For each user, computes |top-t overlap| / t, then averages over users.
    """
    recs_method = np.asarray(recs_method)[:, :t]
    recs_baseline = np.asarray(recs_baseline)[:, :t]
    n_users = recs_method.shape[0]
    overlaps = np.empty(n_users)
    for u in range(n_users):
        overlaps[u] = len(set(recs_method[u]).intersection(recs_baseline[u])) / t
    return float(overlaps.mean())


def subspace_overlap(V_k: np.ndarray, V_hat: np.ndarray) -> float:
    """(1/k) ||V_k^T V_hat||_F^2  in [0, 1]. 1.0 == identical spans.

    Both arguments should have orthonormal columns (shape (n, k)).
    """
    k = V_k.shape[1]
    M = V_k.T @ V_hat
    return float(np.sum(M ** 2) / k)


# ---------------------------------------------------------------------------
# Generic evaluation driver
# ---------------------------------------------------------------------------
def evaluate_method(method_fn,
                    seeds=SEEDS,
                    n_users: int = 200,
                    k: int = K_GROUPS,
                    t: int = 10,
                    rng_seed: int = 12345) -> dict:
    """Score a recommendation method against the SVD baseline across seeds.

    Parameters
    ----------
    method_fn : callable
        method_fn(A, k) -> (recs, V_hat) where
            recs  : (m, >=t) int array of per-user top-t recommendations
            V_hat : (n, k) orthonormal approximate subspace (or None to skip
                    the overlap metric).
        For the baseline self-check, pass `baseline_method`.
    seeds : iterable of int
        Seeds to average over (defaults to the six project seeds).
    n_users : int
        Number of users sampled for the precision average (draft uses 200).
    rng_seed : int
        Seed for choosing WHICH users are sampled (kept fixed across methods
        so comparisons are on the same users).

    Returns
    -------
    dict with keys: precision_mean, precision_std, overlap_mean, overlap_std.
    """
    precisions, overlaps = [], []
    chooser = np.random.default_rng(rng_seed)

    for seed in seeds:
        A = make_matrix(seed)
        m = A.shape[0]
        users = chooser.choice(m, size=min(n_users, m), replace=False)

        base_recs = recommend_all(A, k=k, t=t)[users]
        V_k = fit_subspace(A, k)

        recs, V_hat = method_fn(A, k)
        precisions.append(precision_at_k(recs[users], base_recs, t=t))
        if V_hat is not None:
            overlaps.append(subspace_overlap(V_k, V_hat))

    out = {
        "precision_mean": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions)),
    }
    if overlaps:
        out["overlap_mean"] = float(np.mean(overlaps))
        out["overlap_std"] = float(np.std(overlaps))
    return out


def baseline_method(A: np.ndarray, k: int = K_GROUPS):
    """Reference method: the SVD baseline scored against itself.

    Sanity check — evaluate_method(baseline_method) must give precision 1.0
    and overlap 1.0. Tim's Tang driver should have this exact signature:
        tang_method(A, k) -> (recs, V_hat)
    """
    return recommend_all(A, k=k), fit_subspace(A, k)


if __name__ == "__main__":
    # Self-check: baseline vs. itself must be perfect.
    res = evaluate_method(baseline_method)
    print("Baseline self-check (must be 1.000 / 1.000):")
    print(f"  precision@10 = {res['precision_mean']:.4f} "
          f"+/- {res['precision_std']:.4f}")
    print(f"  overlap      = {res['overlap_mean']:.4f} "
          f"+/- {res['overlap_std']:.4f}")
