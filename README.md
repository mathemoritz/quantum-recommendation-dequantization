# Project code — interface contract

Shared codebase for *Where Does the Quantum Advantage Go? Dequantizing the
Quantum Recommendation Algorithm* (Physics 14N).

**Owner of this layer: Moritz** (Algorithm 1 — the SVD baseline, the data
pipeline, the scoring harness, and Table 1). These modules are the *ground
truth* the rest of the project is measured against.

## Files

| file             | owner  | what it is                                              |
|------------------|--------|---------------------------------------------------------|
| `data.py`        | Moritz | synthetic preference-matrix generator (shared input)    |
| `baseline.py`    | Moritz | classical SVD baseline = ground-truth recommendations   |
| `scoring.py`     | Moritz | precision@10 + subspace-overlap harness (method-agnostic)|
| `make_table1.py` | Moritz | spectrum check + assembles Table 1                      |
| `tang.py`        | **Tim**| Tang's algorithm — **to be written** (see contract below)|

Sandra's KP write-up (Section 4) needs no code — it is conceptual (no quantum
computer). She depends only on the shared **notation in Section 2**, not on
these modules.

## Frozen API (do not change without telling Tim & Sandra)

```python
# data.py
make_matrix(seed) -> A                    # (2000, 600) float64, >= 0, ~rank 8
SEEDS = (0, 1, 2, 3, 4, 5)                # the six averaging seeds
K_GROUPS = 8                              # true rank

# baseline.py
fit_subspace(A, k=8) -> V_k               # (600, 8) TRUE top-k right subspace
recommend_all(A, k=8, t=10) -> R          # (2000, 10) ground-truth top-10/user
singular_values(A) -> s                   # descending, for the spectrum figure

# scoring.py
precision_at_k(recs_method, recs_baseline, t=10) -> float
subspace_overlap(V_k, V_hat) -> float     # (1/k)||V_k^T V_hat||_F^2 in [0,1]
evaluate_method(method_fn, ...) -> dict    # mean/std precision & overlap
```

## The seam — where Tim plugs in

Moritz owns the matrix, the baseline, and the scoring. Tim owns producing
recommendations and an approximate subspace from Tang's algorithm. The single
interface between them is one callable:

```python
# tang.py  (Tim writes this)
def make_tang_method(p, q):
    """Return a method(A, k) -> (recs, V_hat) where
         recs  : (m, >=10) int array, per-user top-10 from Tang's sampler
         V_hat : (600, k) orthonormal approx. of the top-k right subspace (FKV)
    """
    def method(A, k):
        # ... FKV subsample of p rows / q cols, lift to V_hat,
        #     rejection-sample recommendations ...
        return recs, V_hat
    return method
```

Once `tang.py` exists, `python3 make_table1.py` fills the whole table
automatically — the sweep rows come from Tim's method, the anchor row is
Moritz's baseline.

## Handoff checklist (the order things unblock)

1. **`data.py` frozen** → Tim can develop/test Tang on the real input matrix. ✅
2. **`scoring.py` frozen** → Tim can validate Tang against the baseline and
   generate Figures 1–3 + the Table 1 sweep rows. ✅
3. **Section 2 notation circulated** → Sandra (Sec. 4, KP) and Tim (Sec. 5,
   Tang) write their prose on the shared symbols.

## Sanity checks (all currently passing)

- `python3 data.py` — spectrum has a clear elbow after σ₈ (108× gap → k=8 right).
- `python3 baseline.py` — V_k orthonormal; recommend_all is (2000, 10).
- `python3 scoring.py` — baseline vs. itself = **1.000 / 1.000** (must be exact).
- `python3 make_table1.py` — prints the table; baseline anchor row real,
  sweep rows await `tang.py`. Writes `figs/spectrum_check.png`.

## Requirements

`numpy` (required), `matplotlib` (optional, only for the spectrum figure).
