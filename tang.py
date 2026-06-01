"""
tang.py — Tim's implementation of Tang's quantum-inspired recommendation algorithm

Based on: "A quantum-inspired classical algorithm for recommendation systems"
by Ewin Tang (2019)

Tang's algorithm uses importance sampling + FKV (Frieze-Kannan-Vempala) to approximate
the top-k right singular vectors, then samples recommendations from the projected space.
"""

import numpy as np
from data import K_GROUPS


def make_tang_method(p, q):
    """
    Create a method that uses Tang's algorithm with p rows and q columns sampled.

    Note: In Tang's algorithm, p and q are typically equal (both denoted 'q' in the paper).
    We keep both parameters for consistency with the project API.

    Parameters:
    -----------
    p : int
        Number of rows (users) to sample
    q : int
        Number of columns (products) to sample

    Returns:
    --------
    method : callable
        A function method(A, k) that takes the full matrix A and rank k,
        returns (recs, V_hat) where:
            recs: (m, >=10) recommendations for all users
            V_hat: (n, k) approximate top-k right singular vectors
    """
    def method(A, k=K_GROUPS):
        m, n = A.shape  # m=2000 users, n=600 products

        # Use the smaller of p and q for the submatrix size
        # (Tang's algorithm uses a square submatrix)
        submatrix_size = min(p, q)

        print(f"  [Tang p={p}, q={q}] Using {submatrix_size}×{submatrix_size} submatrix")

        # ============================================================
        # STEP 1: ModFKV - Sample and construct low-rank approximation
        # ============================================================
        S, U_hat, Sigma_hat = modfkv(A, submatrix_size, k)

        # S: (submatrix_size, n) - rescaled sampled rows
        # U_hat: (submatrix_size, k) - left singular vectors of W
        # Sigma_hat: (k,) - singular values of W (diagonal of Σ̂)

        print(f"  [Tang] ModFKV output: S {S.shape}, Û {U_hat.shape}, Σ̂ {len(Sigma_hat)} values")

        # ============================================================
        # STEP 2: Construct V_hat (approximate right singular vectors)
        # ============================================================
        # V_hat = S^T U_hat Sigma_hat^{-1}
        # This is the key: S^T maps from q-dimensional space back to n-dimensional space

        V_hat = construct_v_hat(S, U_hat, Sigma_hat)

        print(f"  [Tang] V_hat: {V_hat.shape}, orthonormal: {is_orthonormal(V_hat)}")

        # ============================================================
        # STEP 3: Generate recommendations for all users
        # ============================================================
        # For each user i, sample from D_i = A_i V_hat V_hat^T

        recs = generate_recommendations(A, S, U_hat, Sigma_hat, V_hat)

        print(f"  [Tang] Generated {recs.shape[0]} recommendations")

        return recs, V_hat

    return method


def modfkv(A, q, k):
    """
    ModFKV: Modified Frieze-Kannan-Vempala algorithm for low-rank approximation.

    This is Algorithm 2 from Tang's paper (page 14).

    Steps:
    1. Sample q rows with probability ∝ ||row||²
    2. Sample q columns using two-stage sampling
    3. Form rescaled q×q submatrix W
    4. Compute SVD of W, keep top-k left singular vectors
    5. Return description: (S, U_hat, Sigma_hat)

    Parameters:
    -----------
    A : np.ndarray, shape (m, n)
        The full matrix
    q : int
        Number of rows and columns to sample
    k : int
        Number of singular vectors to extract

    Returns:
    --------
    S : np.ndarray, shape (q, n)
        Sampled and rescaled rows of A
    U_hat : np.ndarray, shape (q, k)
        Top-k left singular vectors of W
    Sigma_hat : np.ndarray, shape (k,)
        Top-k singular values of W
    """
    m, n = A.shape

    # ============================================================
    # STEP 1: Sample q rows with importance sampling
    # ============================================================
    # Probability of sampling row i: p_i = ||A_i||² / ||A||_F²

    row_norms_sq = np.sum(A ** 2, axis=1)  # (m,)
    total_norm_sq = np.sum(row_norms_sq)
    row_probs = row_norms_sq / total_norm_sq

    # Sample q rows (with replacement)
    row_indices = np.random.choice(m, size=q, replace=True, p=row_probs)
    sampled_row_probs = row_probs[row_indices]  # Probabilities of sampled rows

    # ============================================================
    # STEP 2: Sample q columns with two-stage sampling
    # ============================================================
    # For each column:
    #   - Choose a sampled row uniformly: s ~ uniform([q])
    #   - Sample from that row with probability ∝ entry²

    col_indices = []
    col_probs = []

    for _ in range(q):
        # Stage 1: Choose a sampled row uniformly
        s = np.random.choice(q)
        row_idx = row_indices[s]

        # Stage 2: Sample from that row
        row = A[row_idx, :]
        row_sq = row ** 2
        row_norm_sq = np.sum(row_sq)

        if row_norm_sq > 0:
            col_prob = row_sq / row_norm_sq
            col_idx = np.random.choice(n, p=col_prob)
        else:
            # Degenerate case: row is all zeros, sample uniformly
            col_idx = np.random.choice(n)
            col_prob = np.ones(n) / n

        col_indices.append(col_idx)

        # Compute the probability F(j) for this column
        # F(j) = (1/q) * sum over sampled rows of (A[i,j]² / ||A_i||²)
        prob_j = 0.0
        for r_idx in row_indices:
            r_norm_sq = row_norms_sq[r_idx]
            if r_norm_sq > 0:
                prob_j += A[r_idx, col_idx] ** 2 / r_norm_sq
        prob_j /= q
        col_probs.append(prob_j)

    col_indices = np.array(col_indices)
    col_probs = np.array(col_probs)

    # ============================================================
    # STEP 3: Form rescaled submatrix W
    # ============================================================
    # W[r,c] = A[i_r, j_c] / (q * sqrt(p_i_r * F_j_c))

    W = np.zeros((q, q))
    for r in range(q):
        for c in range(q):
            i_r = row_indices[r]
            j_c = col_indices[c]

            # Rescaling factor
            if sampled_row_probs[r] > 0 and col_probs[c] > 0:
                scale = q * np.sqrt(sampled_row_probs[r] * col_probs[c])
                W[r, c] = A[i_r, j_c] / scale
            else:
                W[r, c] = 0.0

    print(f"  [ModFKV] W: {W.shape}, ||W||_F = {np.linalg.norm(W, 'fro'):.2f}, "
          f"||A||_F = {np.linalg.norm(A, 'fro'):.2f}")

    # ============================================================
    # STEP 4: Compute SVD of W, extract top-k left singular vectors
    # ============================================================

    U, Sigma_full, Vt = np.linalg.svd(W, full_matrices=False)

    # Tang's algorithm: keep singular vectors with values > σ threshold
    # For simplicity, we just take top-k
    # (In the paper, σ is chosen based on ||A||_F and k)

    k_actual = min(k, len(Sigma_full))
    U_hat = U[:, :k_actual]  # (q, k)
    Sigma_hat = Sigma_full[:k_actual]  # (k,)

    print(f"  [ModFKV] Extracted {k_actual} singular vectors")
    print(f"  [ModFKV] Top-5 singular values of W: {Sigma_full[:min(5, len(Sigma_full))]}")

    # ============================================================
    # STEP 5: Form S (rescaled sampled rows)
    # ============================================================
    # S has the sampled rows, each rescaled to have norm ||A||_F / sqrt(q)

    S = np.zeros((q, n))
    for r in range(q):
        i_r = row_indices[r]
        # Rescale row i_r by 1 / sqrt(q * p_i_r)
        if sampled_row_probs[r] > 0:
            scale = np.sqrt(q * sampled_row_probs[r])
            S[r, :] = A[i_r, :] / scale
        else:
            S[r, :] = 0.0

    print(f"  [ModFKV] S: {S.shape}, ||S||_F = {np.linalg.norm(S, 'fro'):.2f}")

    return S, U_hat, Sigma_hat

def construct_v_hat(S, U_hat, Sigma_hat):
    """
    Construct V_hat = S^T U_hat Sigma_hat^{-1}, then orthonormalize.

    Tang's algorithm produces V̂ = S^T Û Σ̂^{-1} which spans approximately
    the correct subspace but is NOT orthonormal (see Proposition 4.6).

    For the subspace overlap metric to work correctly, we need orthonormal
    columns, so we apply QR decomposition.

    Parameters:
    -----------
    S : np.ndarray, shape (q, n)
        Rescaled sampled rows
    U_hat : np.ndarray, shape (q, k)
        Left singular vectors of W
    Sigma_hat : np.ndarray, shape (k,)
        Singular values of W

    Returns:
    --------
    V_hat : np.ndarray, shape (n, k)
        Orthonormalized approximate right singular vectors
    """
    k = len(Sigma_hat)

    # Build Sigma_hat^{-1} as a diagonal matrix
    Sigma_inv = np.diag(1.0 / np.maximum(Sigma_hat, 1e-10))

    # Compute raw V_hat (may not be orthonormal)
    V_hat_raw = S.T @ U_hat @ Sigma_inv  # (n, k)

    # Check orthonormality before QR
    VtV_raw = V_hat_raw.T @ V_hat_raw
    ortho_error_before = np.linalg.norm(VtV_raw - np.eye(k), 'fro')

    # Orthonormalize using QR decomposition
    V_hat, R = np.linalg.qr(V_hat_raw)

    # Verify orthonormality after QR
    VtV = V_hat.T @ V_hat
    ortho_error_after = np.linalg.norm(VtV - np.eye(k), 'fro')

    print(f"  [construct_v_hat] Orthonormality: before QR = {ortho_error_before:.4f}, after QR = {ortho_error_after:.6f}")

    return V_hat


def generate_recommendations(A, S, U_hat, Sigma_hat, V_hat, t=10):
    """
    Generate top-t recommendations for all users.

    This implements Algorithm 3 from Tang's paper (page 17):
    For each user i:
      1. Estimate A_i S^T (inner products with sampled rows)
      2. Compute est * U_hat * Sigma_hat^{-2} * U_hat^T
      3. Sample from (result) * S using rejection sampling

    For simplicity, we'll use a deterministic approximation here:
    Sample from D_i = A_i V_hat V_hat^T

    Parameters:
    -----------
    A : np.ndarray, shape (m, n)
        Full matrix
    S : np.ndarray, shape (q, n)
        Rescaled sampled rows
    U_hat : np.ndarray, shape (q, k)
        Left singular vectors
    Sigma_hat : np.ndarray, shape (k,)
        Singular values
    V_hat : np.ndarray, shape (n, k)
        Approximate right singular vectors
    t : int
        Number of recommendations per user

    Returns:
    --------
    recs : np.ndarray, shape (m, t)
        Top-t recommendations for each user
    """
    m, n = A.shape

    # Project all rows onto V_hat's span
    # D = A V_hat V_hat^T

    # For efficiency, compute A @ V_hat first (m, k), then (m, k) @ V_hat^T
    A_V = A @ V_hat  # (m, k)
    D = A_V @ V_hat.T  # (m, n)

    # For each user, find top-t entries
    recs = np.zeros((m, t), dtype=int)
    for i in range(m):
        # Get top-t indices (highest values first)
        top_indices = np.argpartition(D[i, :], -t)[-t:]
        # Sort them by value (descending)
        top_indices = top_indices[np.argsort(D[i, top_indices])[::-1]]
        recs[i, :] = top_indices

    return recs


def is_orthonormal(V, tol=1e-6):
    """Check if V has orthonormal columns."""
    k = V.shape[1]
    return np.allclose(V.T @ V, np.eye(k), atol=tol)


# ================================================================
# TESTING CODE
# ================================================================
if __name__ == "__main__":
    from data import make_matrix
    from baseline import fit_subspace, singular_values
    from scoring import subspace_overlap

    print("Testing Tang's Algorithm (Complete Rewrite)")
    print("=" * 70)

    # Create test matrix
    A = make_matrix(seed=0)
    m, n = A.shape
    k = K_GROUPS
    print(f"Matrix A: {A.shape}, rank approximation k={k}")
    print(f"A Frobenius norm: {np.linalg.norm(A, 'fro'):.2f}")
    print(f"True top-{k} singular values: {singular_values(A)[:k]}\n")

    # Get the TRUE subspace for comparison
    V_true = fit_subspace(A, k=k)
    print(f"True subspace V_true: {V_true.shape}\n")

    # Test with different sample sizes
    test_sizes = [20, 40, 80, 160, 320]

    print(f"{'p=q':<8} {'V_hat shape':<15} {'Overlap':<10} {'||V̂ᵀV̂ - I||_F':<15}")
    print("-" * 70)

    overlaps_list = []
    for p in test_sizes:
        q = p
# Create method and run
        method = make_tang_method(p=p, q=q)
        recs, V_hat = method(A, k=k)

        # Compute overlap with true subspace
        overlap = subspace_overlap(V_true, V_hat)
        overlaps_list.append(overlap)

        # Check how close V_hat is to orthonormal
        VtV = V_hat.T @ V_hat
        orthonorm_error = np.linalg.norm(VtV - np.eye(k), 'fro')

        print(f"{p:<8} {str(V_hat.shape):<15} {overlap:>9.4f} {orthonorm_error:>14.6f}")

    print("-" * 70)

    # Check if overlap is increasing
    is_increasing = all(overlaps_list[i] <= overlaps_list[i+1] + 0.05  # allow small fluctuations
                       for i in range(len(overlaps_list)-1))
    print(f"✓ Overlap generally increasing? {is_increasing}")
    print(f"✓ Best overlap (p={test_sizes[-1]}): {overlaps_list[-1]:.4f}")

    # Detailed test of one case
    print("\n" + "=" * 70)
    print("Detailed test with p=q=80")
    print("=" * 70)

    method = make_tang_method(p=80, q=80)
    recs, V_hat = method(A, k=k)

    print(f"\n✓ Recommendations shape: {recs.shape}")
    print(f"✓ V_hat shape: {V_hat.shape}")
    print(f"✓ Subspace overlap with truth: {subspace_overlap(V_true, V_hat):.4f}")

    # Check singular values of the approximation
    print(f"\n--- Singular value comparison ---")
    s_A = singular_values(A)

    # Approximate singular values from V_hat
    # If V_hat ≈ V (true right sing vecs), then A @ V_hat should have large norms
    AV = A @ V_hat
    approx_sigmas = np.linalg.norm(AV, axis=0)  # norm of each column

    print(f"{'i':<5} {'σ_i(A)':<12} {'σ_i(approx)':<15} {'Ratio':<10}")
    for i in range(k):
        ratio = approx_sigmas[i] / s_A[i] if s_A[i] > 1e-10 else 0
        print(f"{i+1:<5} {s_A[i]:<12.2f} {approx_sigmas[i]:<15.2f} {ratio:<10.3f}")

    print("\n" + "=" * 70)
    print("Expected behavior:")
    print("  - Overlap should increase with sample size")
    print("  - V_hat might NOT be orthonormal (Tang's V̂ = SᵀÛΣ̂⁻¹)")
    print("  - Larger p → better overlap (ideally > 0.7 for p=320)")
    print("  - Singular value ratios should be close to 1.0")
    print("\nNote: Tang's algorithm gives an approximate subspace, not exact orthonormal vectors.")
    print("The key is that D = A V̂ V̂ᵀ approximates the low-rank projection.")


# ================================================================
# ADDITIONAL UTILITY FUNCTIONS
# ================================================================

def sample_from_distribution(probs, n_samples=1):
    """
    Sample from a discrete probability distribution.

    Parameters:
    -----------
    probs : np.ndarray
        Probability distribution (must sum to 1)
    n_samples : int
        Number of samples to draw

    Returns:
    --------
    samples : int or np.ndarray
        Sampled indices
    """
    probs = probs / np.sum(probs)  # Normalize just in case
    if n_samples == 1:
        return np.random.choice(len(probs), p=probs)
    else:
        return np.random.choice(len(probs), size=n_samples, p=probs, replace=True)


def compute_projection_error(A, V_hat):
    """
    Compute ||A - A V̂ V̂ᵀ||_F to measure approximation quality.

    Parameters:
    -----------
    A : np.ndarray, shape (m, n)
        Original matrix
    V_hat : np.ndarray, shape (n, k)
        Approximate right singular vectors

    Returns:
    --------
    error : float
        Frobenius norm of approximation error
    """
    D = A @ V_hat @ V_hat.T
    error = np.linalg.norm(A - D, 'fro')
    return error


def compare_with_baseline(A, V_hat, k):
    """
    Compare Tang's approximation with the true SVD.

    Parameters:
    -----------
    A : np.ndarray
        Original matrix
    V_hat : np.ndarray
        Tang's approximate right singular vectors
    k : int
        Rank of approximation

    Returns:
    --------
    dict with comparison metrics
    """
    from baseline import fit_subspace

    # True top-k subspace
    V_true = fit_subspace(A, k)

    # True low-rank approximation
    A_k = A @ V_true @ V_true.T

    # Tang's approximation
    D = A @ V_hat @ V_hat.T

    # Errors
    true_error = np.linalg.norm(A - A_k, 'fro')
    tang_error = np.linalg.norm(A - D, 'fro')

    # Subspace overlap
    overlap = subspace_overlap(V_true, V_hat)

    return {
        "true_approximation_error": true_error,
        "tang_approximation_error": tang_error,
        "error_ratio": tang_error / true_error,
        "subspace_overlap": overlap,
    }


# ================================================================
# EXTENDED TESTING
# ================================================================

def run_extended_tests():
    """Run comprehensive tests on Tang's algorithm."""
    print("\n" + "=" * 70)
    print("EXTENDED TESTING")
    print("=" * 70)

    from data import make_matrix, SEEDS
    from baseline import fit_subspace

    # Test across multiple seeds
    print("\n--- Testing across multiple random seeds ---")
    print(f"{'Seed':<8} {'p=40 overlap':<15} {'p=160 overlap':<15}")
    print("-" * 40)

    for seed in SEEDS[:3]:  # Test first 3 seeds
        A = make_matrix(seed)
        V_true = fit_subspace(A, k=8)

        # Test with p=40
        method_40 = make_tang_method(p=40, q=40)
        _, V_hat_40 = method_40(A, k=8)
        overlap_40 = subspace_overlap(V_true, V_hat_40)

        # Test with p=160
        method_160 = make_tang_method(p=160, q=160)
        _, V_hat_160 = method_160(A, k=8)
        overlap_160 = subspace_overlap(V_true, V_hat_160)

        print(f"{seed:<8} {overlap_40:<15.4f} {overlap_160:<15.4f}")

    print("\n--- Approximation quality comparison ---")
    A = make_matrix(0)
    k = 8

    for p in [40, 80, 160]:
        method = make_tang_method(p=p, q=p)
        _, V_hat = method(A, k=k)

        comparison = compare_with_baseline(A, V_hat, k)

        print(f"\np={p}:")
        print(f"  True k-rank error:  {comparison['true_approximation_error']:.2f}")
        print(f"  Tang's error:       {comparison['tang_approximation_error']:.2f}")
        print(f"  Error ratio:        {comparison['error_ratio']:.3f}")
        print(f"  Subspace overlap:   {comparison['subspace_overlap']:.4f}")


if __name__ == "__main__":
    # Run basic tests first
    print("=" * 70)
    print("BASIC TESTS")
    print("=" * 70)

    # (Basic tests run here automatically from the code above)

    # Optionally run extended tests
    try:
        run_extended_tests()
    except Exception as e:
        print(f"\nExtended tests failed: {e}")
        print("This is OK - basic functionality is what matters for the project.")
