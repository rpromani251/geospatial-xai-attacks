"""Collapsed Gibbs sampler for a Bernoulli Stochastic Block Model.

Model
-----
Nodes V = {1, ..., n} each have a latent block label z_i in {1, ..., K}.
Edges A_{ij} in {0, 1} are conditionally independent given z:
    A_{ij} | z_i = r, z_j = s  ~  Bernoulli(B_{rs})
with a conjugate prior
    B_{rs}                     ~  Beta(alpha, beta)
and a symmetric Dirichlet prior on the block-proportion vector
    (pi_1, ..., pi_K)          ~  Dirichlet(gamma, ..., gamma)
which implies the marginal
    z_i                        ~  Categorical(pi).

Both B_{rs} and pi are integrated out analytically (Beta-Bernoulli and
Dirichlet-Categorical conjugacy). What remains is a Markov chain on z.

Performance refactor (Phase 1)
------------------------------
The hot path maintains an (n, K) edge-to-block matrix E with
    E[i, s]  =  #edges from i to nodes currently in block s
and updates M, T, and E via O(K) + O(n) vectorized operations per Gibbs step.
The per-node candidate-block scoring uses vectorized gammaln over a (K, K)
posterior-parameter grid, replacing a K^2 Python loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import gammaln
from sklearn.cluster import SpectralClustering


# ── Log Beta-Binomial marginal likelihood (edge block) ──────────────────────

def _log_beta(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """log B(a, b) using log-gamma. Vectorised."""
    return gammaln(a) + gammaln(b) - gammaln(a + b)


def _log_beta_binomial(k: np.ndarray, n: np.ndarray,
                      alpha_post: np.ndarray, beta_post: np.ndarray) -> np.ndarray:
    """log P(k edges | n trials, Beta(alpha_post, beta_post) prior on p).

    Binomial coefficient dropped — cancels across r-candidates.
    """
    return _log_beta(alpha_post + k, beta_post + n - k) - _log_beta(alpha_post, beta_post)


# ── SBM state container ─────────────────────────────────────────────────────

@dataclass
class SBM:
    """Fitted SBM state.

    z       : (n,) int array of block assignments (values 0..K-1)
    K       : number of blocks (fixed)
    M       : (K, K) edge-count matrix, M[r, s] = observed edges between r and s
              (for r==s, M[r,r] = 2 * #edges within r — ordered-pair convention)
    T       : (K, K) trial-count matrix, ordered pairs
    z_trace : list of (n,) partitions recorded post burn-in
    """
    z: np.ndarray
    K: int
    M: np.ndarray
    T: np.ndarray
    z_trace: list

    def co_clustering_matrix(self) -> np.ndarray:
        if not self.z_trace:
            return np.eye(len(self.z))
        stacked = np.stack(self.z_trace)
        n = stacked.shape[1]
        co = np.zeros((n, n), dtype=float)
        for z_s in stacked:
            co += (z_s[:, None] == z_s[None, :]).astype(float)
        co /= len(self.z_trace)
        return co

    def block_probability_matrix(self, alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
        return (self.M + alpha) / (np.maximum(self.T, 0) + alpha + beta)


# ── State initialization ────────────────────────────────────────────────────

def _init_state(A: np.ndarray, z: np.ndarray, K: int):
    """Build E, M, T, block_sizes from adjacency and assignments.

    E[i, s]  =  sum_j A[i, j] * 1[z_j == s]          shape (n, K)
    M[r, s]  =  sum_{i in r} E[i, s]                  shape (K, K)
                (ordered-pair convention: M[r,r] = 2 * #edges within r)
    T[r, s]  =  bs[r] * bs[s]  for r != s, else bs[r] * (bs[r] - 1)
    """
    n = A.shape[0]
    onehot = np.zeros((n, K), dtype=np.int64)
    onehot[np.arange(n), z] = 1
    E = A.astype(np.int64) @ onehot                # (n, K)
    M = onehot.T @ E                                # (K, K)
    block_sizes = np.bincount(z, minlength=K).astype(np.int64)
    T = np.outer(block_sizes, block_sizes).astype(np.int64)
    np.fill_diagonal(T, block_sizes * (block_sizes - 1))
    return E, M, T, block_sizes


# Kept for backward compat with external imports.
def _initial_counts(A: np.ndarray, z: np.ndarray, K: int):
    _, M, T, _ = _init_state(A, z, K)
    return M, T


# ── Vectorized candidate-block scoring ──────────────────────────────────────

def _log_edges_all_r(e_i: np.ndarray,
                     bs_no_i: np.ndarray,
                     M: np.ndarray,
                     T: np.ndarray,
                     alpha: float,
                     beta: float) -> np.ndarray:
    """log P(edges from i | z_i = r, z_{-i}) for every candidate r, vectorized.

    Parameters
    ----------
    e_i    : (K,) edges from i to each current block (= E[i, :] with i's
             block-c contribution already handled by the caller via M, T update).
    bs_no_i: (K,) block sizes with i removed.
    M, T   : current (K, K) count matrices after removing i.

    Returns
    -------
    (K,) array of log-Beta-Binomial sums indexed by candidate r.
    """
    # Posterior Beta parameters per block pair (r, s).
    Mf = M.astype(np.float64)
    Tf = T.astype(np.float64)
    dM = np.diag(Mf).copy()
    dT = np.diag(Tf).copy()

    alpha_post = alpha + Mf                        # off-diag OK
    beta_post = beta + (Tf - Mf)                   # off-diag OK
    np.fill_diagonal(alpha_post, alpha + dM / 2.0) # halve within-block
    np.fill_diagonal(beta_post,  beta + (dT - dM) / 2.0)

    k = e_i.astype(np.float64)[None, :]            # (1, K), broadcast over r
    n = bs_no_i.astype(np.float64)[None, :]        # (1, K)

    # log B(a+k, b+n-k) - log B(a, b), vectorized gammaln on (K, K).
    ln_num = (gammaln(alpha_post + k)
              + gammaln(beta_post + n - k)
              - gammaln(alpha_post + beta_post + n))
    ln_den = (gammaln(alpha_post)
              + gammaln(beta_post)
              - gammaln(alpha_post + beta_post))
    term = ln_num - ln_den                         # (K, K)
    return term.sum(axis=1)                        # (K,)


# ── Incremental remove / add helpers ────────────────────────────────────────

def _remove_from_block(i: int, c: int, A_col_i: np.ndarray, e_i: np.ndarray,
                       E: np.ndarray, M: np.ndarray, T: np.ndarray,
                       block_sizes: np.ndarray) -> None:
    """Update M, T, block_sizes so they exclude node i (currently in block c).

    E and z[i] are left unchanged by this helper; caller tracks e_i = E[i].
    """
    block_sizes[c] -= 1
    # M: subtract row and column e_i; diagonal gets -2*e_i[c] automatically.
    M[c, :] -= e_i
    M[:, c] -= e_i
    # T: subtract bs_no_i row and column; diagonal gets -2*bs_no_i[c].
    T[c, :] -= block_sizes
    T[:, c] -= block_sizes


def _add_to_block(i: int, r: int, A_col_i: np.ndarray, e_i: np.ndarray,
                  E: np.ndarray, M: np.ndarray, T: np.ndarray,
                  block_sizes: np.ndarray, prev_c: int) -> None:
    """Insert node i into block r. block_sizes[r] still excludes i on entry.

    If r != prev_c, also moves the column-c entries in E[:, c] to E[:, r]
    for every neighbor j of i (only neighbors see their edge-to-block counts
    change).
    """
    if r != prev_c:
        # Update E for neighbors of i: the edge from j to i was counted in
        # column prev_c; it should now be in column r.
        E[:, prev_c] -= A_col_i
        E[:, r] += A_col_i
    # M and T updates symmetric to remove, using bs_no_i (i.e. current
    # block_sizes, which still excludes i).
    M[r, :] += e_i
    M[:, r] += e_i
    T[r, :] += block_sizes
    T[:, r] += block_sizes
    block_sizes[r] += 1


# ── Main sampler ─────────────────────────────────────────────────────────────

def fit_sbm(
    A: np.ndarray,
    K: int,
    n_iter: int = 500,
    burn_in: int = 200,
    thin: int = 5,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    z_init: Optional[np.ndarray] = None,
    seed: int = 0,
    verbose: bool = True,
) -> SBM:
    """Collapsed Gibbs sampler for Bernoulli SBM with fixed K.

    See module docstring for the model. Returns a thinned trace of post-burn-in
    partitions.
    """
    assert A.shape[0] == A.shape[1], "adjacency must be square"
    assert np.array_equal(A, A.T), "adjacency must be symmetric"
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    A_int = A.astype(np.int64, copy=False)

    if isinstance(z_init, str) and z_init == "random":
        z = rng.integers(0, K, size=n)
    elif z_init is None:
        sc = SpectralClustering(
            n_clusters=K, affinity="precomputed",
            random_state=seed, assign_labels="kmeans",
        )
        z = sc.fit_predict(A.astype(float)).astype(int)
    else:
        z = z_init.astype(int).copy()

    E, M, T, block_sizes = _init_state(A_int, z, K)
    z_trace: list[np.ndarray] = []

    for it in range(n_iter):
        for i in range(n):
            c = int(z[i])
            A_col_i = A_int[:, i]
            e_i = E[i].copy()   # snapshot; E[i] itself won't change within this node's move

            # Remove i from block c.
            _remove_from_block(i, c, A_col_i, e_i, E, M, T, block_sizes)

            # Score every candidate block r in one vectorized pass.
            log_prior = np.log(block_sizes + gamma)
            log_edges = _log_edges_all_r(e_i, block_sizes, M, T, alpha, beta)
            log_probs = log_prior + log_edges

            log_probs -= log_probs.max()
            probs = np.exp(log_probs)
            probs /= probs.sum()
            r_new = int(rng.choice(K, p=probs))

            z[i] = r_new
            _add_to_block(i, r_new, A_col_i, e_i, E, M, T, block_sizes, prev_c=c)

        if verbose and (it % max(1, n_iter // 10) == 0):
            sizes = np.bincount(z, minlength=K)
            print(f"  iter {it:4d}  block sizes={sizes.tolist()}")

        if it >= burn_in and (it - burn_in) % thin == 0:
            z_trace.append(z.copy())

    return SBM(z=z, K=K, M=M.copy(), T=T.copy(), z_trace=z_trace)
