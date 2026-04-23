"""Build behavioral-similarity graphs from explanation matrices.

Nodes are instances; edge weight = cosine similarity between their attribution
vectors (SHAP or LIME). Sparsified via k-nearest-neighbours and binarised for
Bernoulli SBM consumption.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity_matrix(E: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of rows of E.

    E: (n, d) matrix of attribution vectors (e.g. SHAP values).
    Returns (n, n) similarity matrix with entries in [-1, 1]; diagonal zeroed.
    """
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    En = E / norms
    S = En @ En.T
    np.fill_diagonal(S, 0.0)
    return S


def build_knn_similarity_graph(E: np.ndarray, k: int = 8, symmetric: bool = True) -> np.ndarray:
    """k-NN cosine-similarity graph, binarised.

    For each node, connect to its top-k most cosine-similar neighbours.
    If symmetric=True, symmetrise by OR (edge exists if either endpoint chose
    the other as a neighbour).

    Returns an (n, n) symmetric adjacency matrix with {0, 1} entries and zero
    diagonal.
    """
    n = E.shape[0]
    S = cosine_similarity_matrix(E)

    # Top-k neighbours per row (descending similarity)
    topk = np.argpartition(-S, kth=k, axis=1)[:, :k]

    A = np.zeros((n, n), dtype=np.int8)
    rows = np.repeat(np.arange(n), k)
    cols = topk.ravel()
    A[rows, cols] = 1

    if symmetric:
        A = ((A + A.T) > 0).astype(np.int8)

    np.fill_diagonal(A, 0)
    return A
