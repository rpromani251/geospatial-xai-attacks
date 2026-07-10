"""Partially-Exchangeable SBM (pEx-SBM) — collapsed Gibbs sampler.

Difference from plain SBM (`sbm.py`)
------------------------------------
In plain SBM the prior on the block of each node is a single Categorical:
    z_i | π      ~ Categorical(π)
    π            ~ Dirichlet(γ, ..., γ)

In pEx-SBM we have observed *layer* labels ℓ_i ∈ {1, ..., L} (binary here:
layer-high vs layer-low) and each layer gets its own block-proportion vector:
    z_i | ℓ_i = ℓ, π^(ℓ)  ~ Categorical(π^(ℓ))
    π^(ℓ)                ~ Dirichlet(γ, ..., γ)   iid over ℓ

Block *identities* are shared across layers (same K blocks, same B matrix on
edges); only block *proportions* are allowed to differ by layer. Hence
"partially exchangeable": nodes are exchangeable inside a layer, not across.

Effect on the Gibbs update
--------------------------
Only the Dirichlet-Categorical term changes. For node i with layer ℓ_i:
    p(z_i = r | z_{-i}, ℓ) ∝ (n_{r,ℓ_i}^{(-i)} + γ)
where n_{r,ℓ_i}^{(-i)} = #{j ≠ i : z_j = r, ℓ_j = ℓ_i}.

The Beta-Binomial edge term is unchanged (block matrix B is still global).

Why this should separate better
-------------------------------
If the attack routes behaviour by the layer, a block that is densely
populated in the high-layer but not in the low-layer becomes easy to
identify: the prior rewards putting high-layer nodes into high-layer-dominated
blocks. Plain SBM must rediscover this from edges alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import SpectralClustering

from .sbm import SBM, _initial_counts, _log_beta_binomial


@dataclass
class PExSBM(SBM):
    """Fitted pEx-SBM state.

    Extends SBM with per-layer block-count table.
      N_layer_block : (L, K) int — N_layer_block[ℓ, r] = # nodes in layer ℓ assigned to r
    """
    layer: np.ndarray = None
    N_layer_block: np.ndarray = None


def _layer_block_counts(z: np.ndarray, layer: np.ndarray, K: int, L: int) -> np.ndarray:
    N = np.zeros((L, K), dtype=np.int64)
    for ell in range(L):
        for r in range(K):
            N[ell, r] = int(((layer == ell) & (z == r)).sum())
    return N


def fit_pex_sbm(
    A: np.ndarray,
    layer: np.ndarray,
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
) -> PExSBM:
    """Collapsed Gibbs for pEx-SBM with fixed K and observed layer labels.

    Parameters mirror `fit_sbm`, plus:
      layer : (n,) int — observed layer label per node, values in [0, L-1]
    """
    assert A.shape[0] == A.shape[1], "adjacency must be square"
    assert np.array_equal(A, A.T), "adjacency must be symmetric"
    assert len(layer) == A.shape[0], "layer length must equal n"
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    L = int(layer.max()) + 1

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

    M, T = _initial_counts(A, z, K)
    block_sizes = np.bincount(z, minlength=K).astype(np.int64)
    N_lb = _layer_block_counts(z, layer, K, L)  # (L, K)
    z_trace: list[np.ndarray] = []

    for it in range(n_iter):
        for i in range(n):
            ell_i = int(layer[i])
            c = int(z[i])

            # Remove i from global edge/trial counts and block-size tables
            block_sizes[c] -= 1
            N_lb[ell_i, c] -= 1
            row_i = A[i]
            # Edge/trial removals — replicate _remove_node logic inline
            for s in range(K):
                members_s = np.where(z == s)[0]
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

            # Compute log P(z_i = r | rest, ℓ_i) for each r
            log_probs = np.zeros(K)
            for r in range(K):
                # pEx prior: per-layer count
                log_prior = np.log(N_lb[ell_i, r] + gamma)

                log_edges = 0.0
                for s in range(K):
                    members_s = np.where(z == s)[0]
                    members_s = members_s[members_s != i]
                    t_is = int(block_sizes[s])
                    e_is = int(row_i[members_s].sum())
                    if r == s:
                        alpha_post = alpha + M[r, s] / 2
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

            log_probs -= log_probs.max()
            probs = np.exp(log_probs)
            probs /= probs.sum()
            r_new = int(rng.choice(K, p=probs))

            # Add i back into block r_new
            z[i] = r_new
            for s in range(K):
                members_s = np.where(z == s)[0]
                members_s = members_s[members_s != i]
                e_is = int(row_i[members_s].sum())
                if r_new == s:
                    M[r_new, s] += 2 * e_is
                    T[r_new, s] += 2 * block_sizes[r_new]
                else:
                    M[r_new, s] += e_is
                    M[s, r_new] += e_is
                    T[r_new, s] += block_sizes[s]
                    T[s, r_new] += block_sizes[s]
            block_sizes[r_new] += 1
            N_lb[ell_i, r_new] += 1

        if verbose and (it % max(1, n_iter // 10) == 0):
            sizes = np.bincount(z, minlength=K)
            print(f"  iter {it:4d}  block sizes={sizes.tolist()}  layer-block={N_lb.tolist()}")

        if it >= burn_in and (it - burn_in) % thin == 0:
            z_trace.append(z.copy())

    return PExSBM(
        z=z, K=K, M=M.copy(), T=T.copy(), z_trace=z_trace,
        layer=layer.copy(), N_layer_block=N_lb.copy(),
    )
