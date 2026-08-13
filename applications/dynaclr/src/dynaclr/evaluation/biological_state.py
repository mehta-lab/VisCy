"""Canonical biological-state analysis building blocks.

The supported stack is:

1. per-experiment, per-marker, time-matched control median/MAD normalization;
2. one pooled, marker-specific PCA retaining 80 percent variance;
3. pooled MMD witness scoring with balanced experiment references;
4. one joint control+perturbed, tied-covariance GMM with symmetric abstention;
5. compact causal multi-task TCN training on row-aligned PCA sequences.

Functions are the primary API. YAML and CLI entrypoints are thin wrappers around
these functions.
"""

from dynaclr.evaluation.linear_classifiers.witness_gmm_labels import (
    PooledWitnessGmmFit,
    fit_pooled_witness_gmm,
)
from dynaclr.evaluation.mmd.export_representation import (
    DEFAULT_REPRESENTATION_KEY,
    export_pooled_representation,
)
from dynaclr.evaluation.mmd.representation import (
    ExplainedVariancePCA,
    PreparedMMDRepresentation,
    fit_pca_for_explained_variance,
    prepare_mmd_representation,
)
from dynaclr.evaluation.temporal import (
    CausalSequences,
    CompactTCNConfig,
    MultitaskTCNFit,
    build_causal_sequences,
    fit_multitask_tcn,
)

__all__ = [
    "DEFAULT_REPRESENTATION_KEY",
    "CausalSequences",
    "CompactTCNConfig",
    "ExplainedVariancePCA",
    "MultitaskTCNFit",
    "PooledWitnessGmmFit",
    "PreparedMMDRepresentation",
    "build_causal_sequences",
    "export_pooled_representation",
    "fit_multitask_tcn",
    "fit_pca_for_explained_variance",
    "fit_pooled_witness_gmm",
    "prepare_mmd_representation",
]
