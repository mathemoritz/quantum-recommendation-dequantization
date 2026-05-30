# Physics 14N - Project code

## Files

| file             | owner  | what it is                                              |
|------------------|--------|---------------------------------------------------------|
| `data.py`        | Moritz | synthetic preference-matrix generator (shared input)    |
| `baseline.py`    | Moritz | classical SVD baseline = ground-truth recommendations   |
| `scoring.py`     | Moritz | precision@10 + subspace-overlap harness (method-agnostic)|
| `make_table1.py` | Moritz | spectrum check + assembles Table 1                      |
| `tang.py`        | Tim    | Tang's algorithm                                         |

Sandra's KP write-up has no code, it is conceptual (no quantum
computer).

## API

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

## For Tim

Moritz creates the matrix, the SVD-baseline, and the scoring.
Tim produces the recommendations and an approximate subspace from Tang's algorithm.
For this use:

```python
# tang.py
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

Once you write `tang.py`, I created `python3 make_table1.py` which fills the whole table with the sweep rows come from Tim's method, the anchor row is
Moritz's baseline.

## Order of Tasks

1. **`data.py` frozen** → Tim can develop/test Tang on the real input matrix. ✅
2. **`scoring.py` frozen** → Tim can validate Tang against the baseline and
   generate Figures 1–3 + the Table 1 sweep rows. ✅
3. **Section 2 notation circulated** → Sandra (Sec. 4, KP) and Tim (Sec. 5,
   Tang) write their prose on the shared symbols.

## Sanity checks :)

- `python3 data.py` — spectrum has a clear elbow after SingVal8 (108× gap → k=8 right).
- `python3 baseline.py` — V_k orthonormal; recommend_all is (2000, 10).
- `python3 scoring.py` — baseline vs. itself = **1.000 / 1.000** (must be exact).
- `python3 make_table1.py` — prints the table; baseline anchor row real,
  sweep rows await `tang.py`. Writes `figs/spectrum_check.png`.

## Requirements

`numpy` (required), `matplotlib` (optional, only for the spectrum figure).
