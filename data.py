"""
data.py — Synthetic preference-matrix generator

Public API
    make_matrix(seed) -> A                      # (m, n) float64 ndarray
    make_matrix(seed, return_factors=True) -> (A, info)   # info dict for debugging
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Frozen project constants. Import these everywhere instead of hard-coding.
# ---------------------------------------------------------------------------
M_USERS = 2000      # number of users (rows)
N_PRODUCTS = 600    # number of products (columns)
K_GROUPS = 8        # number of latent interest groups (true rank of signal)
NOISE_LEVEL = 0.05  # additive noise as a fraction of signal scale
SEEDS = (0, 1, 2, 3, 4, 5)  # the six seeds used for averaged results


def make_matrix(
    seed: int,
    m: int = M_USERS,
    n: int = N_PRODUCTS,
    k: int = K_GROUPS,
    noise_level: float = NOISE_LEVEL,
    return_factors: bool = False,
):
    """Generate a synthetic, approximately rank-k, non-negative preference matrix.

    Parameters
    ----------
    seed : int
        RNG seed. make_matrix(seed) is fully deterministic.
    m, n, k : int
        Users, products, latent groups. Defaults are the project values.
    noise_level : float
        Std of additive noise as a fraction of the clean signal's std.
    return_factors : bool
        If True, also return a dict with the planted factors and the
        noiseless matrix, useful for sanity checks and debugging.

    Returns
    -------
    A : np.ndarray, shape (m, n)
        The (noisy, non-negative) preference matrix.
    info : dict   (only if return_factors=True)
        {
          'A_clean': (m, n) noiseless rank-<=k matrix,
          'U': (m, k) user-factor matrix,
          'W': (n, k) product-factor matrix,
          'user_group': (m,) primary group index per user,
        }
    """
    rng = np.random.default_rng(seed)

    # --- 1. Each user belongs MOSTLY to one group -------------------------
    # Primary group per user, then a one-hot-ish factor row with small
    # spillover into other groups so tastes are differentiated but not
    # perfectly orthogonal.
    user_group = rng.integers(0, k, size=m)
    U = 0.10 * rng.random((m, k))                 # small background affinity
    U[np.arange(m), user_group] += 1.0            # dominant primary interest
    # a little secondary interest for ~30% of users to enrich the structure
    has_secondary = rng.random(m) < 0.30
    sec_group = rng.integers(0, k, size=m)
    U[has_secondary, sec_group[has_secondary]] += 0.5

    # --- 2. Each product appeals to ONE OR TWO groups ---------------------
    W = np.zeros((n, k))
    for j in range(n):
        n_groups = 1 if rng.random() < 0.6 else 2
        groups = rng.choice(k, size=n_groups, replace=False)
        W[j, groups] = 0.5 + 0.5 * rng.random(n_groups)  # appeal strength in [0.5, 1]

    # --- 3. Clean low-rank signal (exactly rank <= k) ---------------------
    A_clean = U @ W.T                              # (m, n), >= 0

    # --- 4. Additive noise, scaled to ~noise_level of signal, non-negative
    sigma = noise_level * A_clean.std()
    A = A_clean + sigma * rng.standard_normal((m, n))
    np.clip(A, 0.0, None, out=A)                   # keep entries non-negative

    if return_factors:
        info = {
            "A_clean": A_clean,
            "U": U,
            "W": W,
            "user_group": user_group,
        }
        return A, info
    return A


if __name__ == "__main__":
    # Quick self-check: show that the spectrum has a clear gap after k=8.
    A = make_matrix(0)
    s = np.linalg.svd(A, compute_uv=False)
    print(f"matrix shape: {A.shape}, min entry: {A.min():.3f} (should be >= 0)")
    print("top 12 singular values:")
    for i, val in enumerate(s[:12], start=1):
        marker = "   <-- gap expected after this" if i == K_GROUPS else ""
        print(f"  sigma_{i:2d} = {val:8.2f}{marker}")
    gap_ratio = s[K_GROUPS - 1] / s[K_GROUPS]
    print(f"\nsigma_{K_GROUPS}/sigma_{K_GROUPS+1} = {gap_ratio:.2f} "
          f"(a clear elbow means >> 1)")
