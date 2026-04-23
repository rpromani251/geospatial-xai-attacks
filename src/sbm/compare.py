"""Compare two SBM fits (honest vs adversarial explanation graphs).

Provides:
- Block-probability matrix ψ_{rs} from a fitted SBM
- Layer-conditional statistics (how well a block partition aligns with
  pre-known layer labels like demographic group)
- Partition comparison metrics (NMI, ARI) between two fitted z vectors
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .sbm import SBM


# ── Block-probability matrix ψ ──────────────────────────────────────────────

def block_probability_matrix(sbm: SBM, alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
    """Posterior-mean B_{rs} for each block pair: (edges + alpha) / (trials + alpha + beta)."""
    return sbm.block_probability_matrix(alpha=alpha, beta=beta)


# ── Partition comparison ─────────────────────────────────────────────────────

def partition_nmi(z1: np.ndarray, z2: np.ndarray) -> float:
    """Normalised mutual information between two partitions. Range [0, 1]."""
    return float(normalized_mutual_info_score(z1, z2))


def partition_ari(z1: np.ndarray, z2: np.ndarray) -> float:
    """Adjusted Rand index. 1 = identical, 0 = random-level agreement."""
    return float(adjusted_rand_score(z1, z2))


# ── Layer-conditional block stats ────────────────────────────────────────────

def layer_block_contingency(z: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Count table N[l, r] = # nodes with layer label l in block r.

    If the SBM's block partition aligns with the audit-relevant layer, this
    contingency table is sparse (each layer occupies only a few blocks).
    Uniform entries → block structure ignores the layer.
    """
    K = int(z.max()) + 1
    L = int(layer.max()) + 1
    N = np.zeros((L, K), dtype=np.int64)
    for li in range(L):
        mask = (layer == li)
        zs = z[mask]
        for r in range(K):
            N[li, r] = int((zs == r).sum())
    return N


def layer_separation(z: np.ndarray, layer: np.ndarray) -> float:
    """How much block structure respects the layer labels.

    Returns NMI(z, layer). 1.0 means blocks and layers are identical; 0.0 means
    blocks ignore the layer. High values say "the model's behavioural block
    structure lines up with the audit-relevant partition" — which in our
    setting is the *attack-sensitive* signal.
    """
    return partition_nmi(z, layer)


def layer_conditional_density(A: np.ndarray, z: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """For each (layer_l, block_r) pair, report mean within-edge density.

    Interprets "do members of layer l inside block r connect to each other?"
    Used to spot blocks dominated by a single audit layer.
    """
    L = int(layer.max()) + 1
    K = int(z.max()) + 1
    D = np.zeros((L, K), dtype=float)
    for li in range(L):
        for r in range(K):
            members = np.where((layer == li) & (z == r))[0]
            m = len(members)
            if m < 2:
                D[li, r] = np.nan
                continue
            D[li, r] = A[np.ix_(members, members)].sum() / (m * (m - 1))
    return D


# ── Co-clustering probability summaries ─────────────────────────────────────

def layer_coclustering(co: np.ndarray, layer: np.ndarray) -> tuple[float, float]:
    """Mean co-clustering probability within-layer vs across-layer.

    co      : (n, n) co-clustering matrix P(z_i = z_j) from SBM trace
    layer   : (n,) layer labels

    Returns (within_mean, across_mean).
    Intuition: if the attack breaks explanation structure, within-layer
    co-clustering should collapse toward the across-layer baseline; if the
    attack exaggerates, within grows and across shrinks.
    """
    n = len(layer)
    same = layer[:, None] == layer[None, :]
    # mask the diagonal
    diag = np.eye(n, dtype=bool)
    within = co[same & ~diag]
    across = co[~same]
    return float(within.mean()), float(across.mean())
