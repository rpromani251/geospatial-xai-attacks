"""Finite-K Dirichlet-Categorical pEx-SBM — collapsed Gibbs sampler.

NAMING
------
This is our **finite-K Dirichlet-Categorical** implementation of a
partially-exchangeable SBM: K is fixed, each layer gets its own
Dirichlet(γ) over the same K blocks, and the block proportions are
integrated out against a Beta-Bernoulli edge likelihood.

This is **not** the pEx-SBM of Durante et al. (2025,
arXiv:2410.10619 — "Partially Exchangeable Stochastic Block Models
for (Node-Colored) Multilayer Networks"), which is the H-DP pEx with
a two-level Dirichlet Process hierarchy, K inferred, and a
three-route Gibbs sampler (join-in-layer / bring-in-from-base /
new-profile). See `docs/plain-sbm-poc.md` §15 Fix 4, the Durante
reference sampler at `papers/pexsbm/pex-sbm.R`, and the naming box
in `docs/plain-sbm-poc.md`.

Every null result across the α/γ/noisy/subpop/held-out/penalty
stress tests (`docs/plain-sbm-poc.md` §§9–13) is specific to the
finite-K variant implemented here. H-DP pEx may behave differently
on the same graphs and has not yet been run.

Difference from plain SBM (`sbm.py`)
------------------------------------
Plain SBM has a single Categorical prior on block membership. pEx-SBM has an
observed *layer* label ℓ_i ∈ {1, ..., L} per node, and each layer gets its own
block-proportion vector:
    z_i | ℓ_i = ℓ, π^(ℓ)  ~ Categorical(π^(ℓ))
    π^(ℓ)                 ~ Dirichlet(γ, ..., γ)   iid over ℓ

Block identities (and the edge matrix B) are shared across layers; only block
proportions differ by layer. The Gibbs update changes only in the prior term:
    log P(z_i = r | ...)  ∝  log(N_{ℓ_i, r}^{(-i)} + γ) + log P(edges | ...)

The Beta-Binomial edge term is identical to plain SBM and is shared via
`_log_edges_all_r` in `sbm.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import SpectralClustering

from .sbm import (
    SBM,
    _init_state,
    _initial_counts,          # re-export for callers
    _log_beta_binomial,       # re-export
    _log_edges_all_r,
    _remove_from_block,
    _add_to_block,
)


@dataclass
class PExSBM(SBM):
    """Fitted pEx-SBM state. Adds per-layer block-count table.

    N_layer_block : (L, K) int, N_layer_block[ℓ, r] = # nodes in layer ℓ assigned to r
    """
    layer: np.ndarray = None
    N_layer_block: np.ndarray = None


def _layer_block_counts(z: np.ndarray, layer: np.ndarray, K: int, L: int) -> np.ndarray:
    """Count nodes with observed (layer != -1) labels. -1 entries are ignored."""
    N = np.zeros((L, K), dtype=np.int64)
    for ell in range(L):
        mask = (layer == ell)
        N[ell] = np.bincount(z[mask], minlength=K)
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
    """Collapsed Gibbs for pEx-SBM with fixed K and observed layer labels."""
    assert A.shape[0] == A.shape[1], "adjacency must be square"
    assert np.array_equal(A, A.T), "adjacency must be symmetric"
    assert len(layer) == A.shape[0], "layer length must equal n"
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    # layer entries of -1 are treated as unobserved (fall back to plain prior).
    observed = layer >= 0
    L = int(layer[observed].max()) + 1 if observed.any() else 1
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
    N_lb = _layer_block_counts(z, layer, K, L)  # (L, K)
    z_trace: list[np.ndarray] = []

    for it in range(n_iter):
        for i in range(n):
            ell_i = int(layer[i])
            c = int(z[i])
            A_col_i = A_int[:, i]
            e_i = E[i].copy()

            # Remove i from global edge/trial counts and (if observed) layer-block table.
            _remove_from_block(i, c, A_col_i, e_i, E, M, T, block_sizes)
            if ell_i >= 0:
                N_lb[ell_i, c] -= 1

            # Prior: per-layer count if observed, else plain Dirichlet-Categorical.
            if ell_i >= 0:
                log_prior = np.log(N_lb[ell_i].astype(np.float64) + gamma)
            else:
                log_prior = np.log(block_sizes.astype(np.float64) + gamma)
            log_edges = _log_edges_all_r(e_i, block_sizes, M, T, alpha, beta)
            log_probs = log_prior + log_edges

            log_probs -= log_probs.max()
            probs = np.exp(log_probs)
            probs /= probs.sum()
            r_new = int(rng.choice(K, p=probs))

            z[i] = r_new
            _add_to_block(i, r_new, A_col_i, e_i, E, M, T, block_sizes, prev_c=c)
            if ell_i >= 0:
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
