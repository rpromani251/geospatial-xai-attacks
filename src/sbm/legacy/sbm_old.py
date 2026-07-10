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

Collapsed representation
------------------------
Both B_{rs} (one per block pair) and pi (the block proportions) are integrated
out analytically using Beta-Bernoulli and Dirichlet-Categorical conjugacy.
What remains is a single Markov chain on the n discrete labels z_1, ..., z_n.

Per-node update
---------------
For node i currently in block c, re-sample z_i conditional on all other labels
z_{-i}:

    P(z_i = r | z_{-i}, A)
        =  P(z_i = r | z_{-i})  *  P(A_i | z_i = r, z_{-i})
        proportional to
           (n_r + gamma)                                        # Dirichlet term
         * prod_{s} BetaBinom(e_{is}; t_{is}, alpha + M_{rs}, beta + T_{rs} - M_{rs})

with
    n_r    = # nodes currently in block r  (excluding i)
    e_{is} = # edges from i to block s
    t_{is} = # potential edges from i to block s (= n_s, or n_s - 1 when r == s)
    M_{rs} = # edges between blocks r and s (excluding all edges touching i)
    T_{rs} = # potential edges between blocks r and s (excluding all ... touching i)

The BetaBinom terms arise from integrating B_{rs} out under its Beta(alpha,beta)
prior: prior Beta * likelihood Bernoulli = posterior Beta * Beta-Binomial
marginal likelihood of the edges touching node i.

We work in log space throughout to avoid underflow for large block counts.
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

    We ignore the binomial coefficient because it cancels across r-candidates
    (it depends only on (e_{is}, t_{is}), not on r).
    """
    return _log_beta(alpha_post + k, beta_post + n - k) - _log_beta(alpha_post, beta_post)


# ── SBM state container + sampler ────────────────────────────────────────────

@dataclass
class SBM:
    """Fitted SBM state.

    z       : (n,) int array of block assignments (values 0..K-1)
    K       : number of blocks (fixed)
    M       : (K, K) edge-count matrix, M[r, s] = observed edges between r and s
    T       : (K, K) trial-count matrix, T[r, s] = possible edges between r and s
    z_trace : list of (n,) partitions recorded post burn-in (for marginal stats)
    """
    z: np.ndarray
    K: int
    M: np.ndarray
    T: np.ndarray
    z_trace: list

    def co_clustering_matrix(self) -> np.ndarray:
        """P(z_i = z_j) estimated as the frequency that nodes co-cluster across
        recorded iterations. Shape (n, n), diagonal = 1.
        """
        if not self.z_trace:
            return np.eye(len(self.z))
        stacked = np.stack(self.z_trace)  # (S, n)
        n = stacked.shape[1]
        co = np.zeros((n, n), dtype=float)
        for z_s in stacked:
            co += (z_s[:, None] == z_s[None, :]).astype(float)
        co /= len(self.z_trace)
        return co

    def block_probability_matrix(self, alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
        """Posterior-mean estimate of B_{rs} given the current assignment:
            E[B_{rs} | z, A] = (M[r,s] + alpha) / (T[r,s] + alpha + beta)
        """
        return (self.M + alpha) / (np.maximum(self.T, 0) + alpha + beta)


# ── Count-matrix helpers ─────────────────────────────────────────────────────

def _initial_counts(A: np.ndarray, z: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Build edge-count M and trial-count T matrices given assignments z."""
    n = len(z)
    M = np.zeros((K, K), dtype=np.int64)
    T = np.zeros((K, K), dtype=np.int64)

    # Block sizes
    counts = np.bincount(z, minlength=K)

    for r in range(K):
        for s in range(K):
            if r == s:
                T[r, s] = counts[r] * (counts[r] - 1)  # ordered pairs
            else:
                T[r, s] = counts[r] * counts[s]

    # Edge counts
    for r in range(K):
        rows = np.where(z == r)[0]
        for s in range(K):
            cols = np.where(z == s)[0]
            M[r, s] = A[np.ix_(rows, cols)].sum()

    return M, T


def _remove_node(A: np.ndarray, z: np.ndarray, i: int,
                 M: np.ndarray, T: np.ndarray, block_sizes: np.ndarray) -> None:
    """Remove node i's contribution from M, T, block_sizes (in-place).

    After this call z[i] is left unchanged (caller sets it via _add_node), but
    block_sizes, M, T reflect the graph with i removed.
    """
    c = z[i]
    block_sizes[c] -= 1  # block_sizes[c] now excludes i
    K = len(block_sizes)
    row_i = A[i]
    for s in range(K):
        members_s = np.where(z == s)[0]
        # Always drop i from its own block's member list
        members_s = members_s[members_s != i]
        e_is = int(row_i[members_s].sum())
        if c == s:
            M[c, s] -= 2 * e_is
            T[c, s] -= 2 * block_sizes[c]
        else:
            M[c, s] -= e_is
            M[s, c] -= e_is
            T[c, s] -= block_sizes[s]
            T[s, c] -= block_sizes[s]


def _add_node(A: np.ndarray, z: np.ndarray, i: int, r: int,
              M: np.ndarray, T: np.ndarray, block_sizes: np.ndarray) -> None:
    """Insert node i into block r (in-place update of M, T, block_sizes, z[i]).

    Pre: _remove_node was called, so block_sizes/M/T exclude i.
    """
    z[i] = r
    K = len(block_sizes)
    row_i = A[i]
    for s in range(K):
        members_s = np.where(z == s)[0]
        members_s = members_s[members_s != i]
        e_is = int(row_i[members_s].sum())
        if r == s:
            M[r, s] += 2 * e_is
            T[r, s] += 2 * block_sizes[r]  # block_sizes[r] still excludes i
        else:
            M[r, s] += e_is
            M[s, r] += e_is
            T[r, s] += block_sizes[s]
            T[s, r] += block_sizes[s]
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

    Parameters
    ----------
    A : (n, n) binary, symmetric, zero-diagonal adjacency
    K : number of blocks (fixed)
    alpha, beta : Beta prior hyperparameters on each B_{rs} (edge prob per block pair)
    gamma : symmetric Dirichlet prior on the block-proportion vector
    z_init : initial assignment; if None, use spectral clustering warm-start
        (default, avoids local optima) or pass np.array for custom init.
        Pass "random" as a string to use random uniform init instead.

    Returns
    -------
    SBM  with final z and a thinned trace of post-burn-in partitions.
    """
    assert A.shape[0] == A.shape[1], "adjacency must be square"
    assert np.array_equal(A, A.T), "adjacency must be symmetric"
    rng = np.random.default_rng(seed)
    n = A.shape[0]

    if isinstance(z_init, str) and z_init == "random":
        z = rng.integers(0, K, size=n)
    elif z_init is None:
        # Spectral warm-start: cluster the symmetric adjacency into K groups
        sc = SpectralClustering(
            n_clusters=K, affinity="precomputed",
            random_state=seed, assign_labels="kmeans",
        )
        z = sc.fit_predict(A.astype(float)).astype(int)
    else:
        z = z_init.astype(int).copy()

    M, T = _initial_counts(A, z, K)
    block_sizes = np.bincount(z, minlength=K).astype(np.int64)
    z_trace: list[np.ndarray] = []

    for it in range(n_iter):
        for i in range(n):
            # Remove i's contribution
            _remove_node(A, z, i, M, T, block_sizes)

            # Compute log P(z_i = r | rest) for each candidate block r
            row_i = A[i]
            log_probs = np.zeros(K)
            for r in range(K):
                # Dirichlet-Categorical term for z_i = r
                log_prior = np.log(block_sizes[r] + gamma)

                # Beta-Binomial terms for edges from i under assignment r
                log_edges = 0.0
                for s in range(K):
                    members_s = np.where(z == s)[0]
                    members_s = members_s[members_s != i]
                    t_is = int(block_sizes[s] if s != r else max(block_sizes[s], 0))
                    # when the candidate block is r, and s == r, then i's
                    # potential partners in r is block_sizes[r] (the others
                    # already in r before we add i)
                    e_is = int(row_i[members_s].sum())
                    # Prior-updated Beta parameters for block pair (r, s):
                    if r == s:
                        alpha_post = alpha + M[r, s] / 2  # undirected: halve
                        beta_post = beta + (T[r, s] - M[r, s]) / 2
                    else:
                        alpha_post = alpha + M[r, s]
                        beta_post = beta + T[r, s] - M[r, s]
                    log_edges += _log_beta_binomial(
                        k=np.array(e_is, dtype=float),
                        n=np.array(t_is, dtype=float),
                        alpha_post=np.array(alpha_post, dtype=float),
                        beta_post=np.array(beta_post, dtype=float),
                    )
                log_probs[r] = log_prior + log_edges

            # Sample z_i from categorical(softmax(log_probs))
            log_probs -= log_probs.max()
            probs = np.exp(log_probs)
            probs /= probs.sum()
            r_new = int(rng.choice(K, p=probs))

            # Add i to new block r_new
            _add_node(A, z, i, r_new, M, T, block_sizes)

        if verbose and (it % max(1, n_iter // 10) == 0):
            sizes = np.bincount(z, minlength=K)
            print(f"  iter {it:4d}  block sizes={sizes.tolist()}")

        if it >= burn_in and (it - burn_in) % thin == 0:
            z_trace.append(z.copy())

    return SBM(z=z, K=K, M=M.copy(), T=T.copy(), z_trace=z_trace)
