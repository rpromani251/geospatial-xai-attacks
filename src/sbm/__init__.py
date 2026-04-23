"""Behavioral SBM module — fit SBMs to explanation-similarity graphs."""

from .similarity import build_knn_similarity_graph, cosine_similarity_matrix
from .sbm import SBM, fit_sbm
from .pex_sbm import PExSBM, fit_pex_sbm
from .compare import block_probability_matrix, partition_nmi, partition_ari

__all__ = [
    "build_knn_similarity_graph",
    "cosine_similarity_matrix",
    "SBM",
    "fit_sbm",
    "PExSBM",
    "fit_pex_sbm",
    "block_probability_matrix",
    "partition_nmi",
    "partition_ari",
]
