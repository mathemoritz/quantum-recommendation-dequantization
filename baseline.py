"""
baseline.py — Classical SVD baseline
"""

from __future__ import annotations

import numpy as np

from data import K_GROUPS


def fit_subspace(A: np.ndarray, k: int = K_GROUPS) -> np.ndarray:
    """Return V_k, the top-k right singular vectors of A as columns (n, k).

    This is the TRUE subspace span(V_k). Tang's algorithm produces an
    approximation V_hat to this; scoring.subspace_overlap compares them.
    """
    # full_matrices=False gives the economy SVD; Vt rows are right sing. vecs.
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    V_k = Vt[:k, :].T                 # (n, k), columns are v_1..v_k
    return V_k


def singular_values(A: np.ndarray) -> np.ndarray:
    """All singular values of A, descending. Used for the spectrum figure
    (the 'true spectrum' curve in Tim's Figure 2)."""
    return np.linalg.svd(A, compute_uv=False)


def denoise_row(A_row: np.ndarray, V_k: np.ndarray) -> np.ndarray:
    """Project one user's row onto span(V_k): x_hat = A_row @ V_k @ V_k^T."""
    return (A_row @ V_k) @ V_k.T


def topk(A: np.ndarray, i: int, k: int = K_GROUPS, t: int = 10,
         V_k: np.ndarray | None = None) -> np.ndarray:
    """Top-t recommended product indices for user i, highest denoised first.

    Pass a precomputed V_k to avoid redecomposing A on every call.
    """
    if V_k is None:
        V_k = fit_subspace(A, k)
    x_hat = denoise_row(A[i, :], V_k)
    # argpartition for the top-t, then sort those t by value (descending).
    top = np.argpartition(x_hat, -t)[-t:]
    return top[np.argsort(x_hat[top])[::-1]]


def recommend_all(A: np.ndarray, k: int = K_GROUPS, t: int = 10) -> np.ndarray:
    """(m, t) matrix of top-t recommendations for every user.

    Decomposes A once, then scores all rows at once. This is the ground-truth
    recommendation set that scoring.precision_at_k compares Tang against.
    """
    V_k = fit_subspace(A, k)
    A_hat = (A @ V_k) @ V_k.T                  # (m, n) all denoised rows
    # top-t per row: argpartition then per-row sort
    part = np.argpartition(A_hat, -t, axis=1)[:, -t:]
    rows = np.arange(A.shape[0])[:, None]
    order = np.argsort(A_hat[rows, part], axis=1)[:, ::-1]
    return part[rows, order]


if __name__ == "__main__":
    from data import make_matrix

    A = make_matrix(0)
    V_k = fit_subspace(A)
    print(f"V_k shape: {V_k.shape} (should be ({A.shape[1]}, {K_GROUPS}))")
    print(f"V_k orthonormal? V_k^T V_k ~= I: "
          f"{np.allclose(V_k.T @ V_k, np.eye(K_GROUPS), atol=1e-8)}")
    recs = topk(A, i=0, V_k=V_k)
    print(f"user 0 top-10 products: {recs.tolist()}")
    R = recommend_all(A)
    print(f"recommend_all shape: {R.shape} (should be ({A.shape[0]}, 10))")
