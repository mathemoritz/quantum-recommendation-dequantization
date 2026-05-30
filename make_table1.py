"""
make_table1.py — Assemble Table 1 and the spectrum sanity-check figure

Just run:  python3 make_table1.py
Output: prints the table, writes figs/spectrum_check.png
"""

from __future__ import annotations

import os
import numpy as np

from data import make_matrix, SEEDS, K_GROUPS, M_USERS, N_PRODUCTS
from baseline import singular_values
from scoring import evaluate_method, baseline_method

# The sample budgets swept in the draft (q = p).
P_SWEEP = [20, 40, 80, 160, 320, 640]


def spectrum_check(seed: int = 0, outdir: str = "figs"):
    """Confirm the k=8 elbow and save a spectrum plot (true spectrum curve)."""
    A = make_matrix(seed)
    s = singular_values(A)
    os.makedirs(outdir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.arange(1, 21), s[:20], "o-", color="#3b5b92")
        ax.axvline(K_GROUPS + 0.5, ls="--", color="0.6",
                   label=f"truncation k = {K_GROUPS}")
        ax.set_xlabel("singular value index")
        ax.set_ylabel(r"$\sigma_\ell$")
        ax.set_title("Spectrum of the synthetic matrix (true spectrum)")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(outdir, "spectrum_check.png")
        fig.savefig(path, dpi=150)
        print(f"wrote {path}")
    except ImportError:
        print("matplotlib not installed; skipping figure (numbers below).")
    gap = s[K_GROUPS - 1] / s[K_GROUPS]
    print(f"sigma_{K_GROUPS}/sigma_{K_GROUPS+1} = {gap:.1f}  (clear elbow if >> 1)")


def build_table():
    """Print Table 1. Baseline row is real; sweep rows need Tim's tang_method."""
    # Try to import Tim's implementation; degrade gracefully if absent.
    try:
        from tang import make_tang_method  # Tim provides this
        have_tang = True
    except Exception:
        have_tang = False

    print(f"\n{'p':>6} {'submatrix':>14} {'precision@10':>14} {'overlap':>10}")
    print("-" * 48)

    if have_tang:
        for p in P_SWEEP:
            method = make_tang_method(p=p, q=p)   # -> method(A, k) -> (recs, V_hat)
            r = evaluate_method(method)
            sub = f"{p}x{min(p, N_PRODUCTS)}"
            print(f"{p:>6} {sub:>14} {r['precision_mean']:>14.3f} "
                  f"{r['overlap_mean']:>10.4f}")
    else:
        for p in P_SWEEP:
            sub = f"{p}x{min(p, N_PRODUCTS)}"
            print(f"{p:>6} {sub:>14} {'(Tim: tang.py)':>14} {'--':>10}")

    # --- Moritz's anchor row: the full SVD baseline, perfect by definition.
    base = evaluate_method(baseline_method)
    print("-" * 48)
    print(f"{'full':>6} {f'{M_USERS}x{N_PRODUCTS}':>14} "
          f"{base['precision_mean']:>14.3f} {base['overlap_mean']:>10.4f}")


if __name__ == "__main__":
    print("=== Spectrum sanity check ===")
    spectrum_check()
    print("\n=== Table 1 ===")
    build_table()
