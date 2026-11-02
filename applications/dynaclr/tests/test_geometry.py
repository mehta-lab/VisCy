"""Tests for frozen-embedding geometry diagnostics."""

import numpy as np

from dynaclr.evaluation.geometry import embedding_geometry, grouped_binary_probe, knn_label_purity


def test_effective_rank_distinguishes_isotropic_and_collapsed_geometry():
    rng = np.random.default_rng(7)
    isotropic = rng.normal(size=(1024, 32))
    collapsed = isotropic[:, :1] @ np.ones((1, 32))

    isotropic_metrics = embedding_geometry(isotropic, pairwise_sample=256)
    collapsed_metrics = embedding_geometry(collapsed, pairwise_sample=256)

    assert isotropic_metrics["effective_rank"] > 25
    assert collapsed_metrics["effective_rank"] < 1.01
    assert isotropic_metrics["rank_99"] > collapsed_metrics["rank_99"]


def test_knn_purity_and_grouped_probe_detect_separated_labels():
    rng = np.random.default_rng(11)
    labels = np.repeat(["control", "treated"], 120)
    groups = np.tile(np.repeat(np.arange(6), 20), 2).astype(str)
    features = rng.normal(scale=0.25, size=(240, 8))
    features[labels == "treated", 0] += 3.0

    purity = knn_label_purity(features, labels, n_neighbors=5)
    probe = grouped_binary_probe(features, labels, groups, n_splits=3)

    assert purity["normalized_enrichment"] > 0.8
    assert probe["balanced_accuracy_mean"] > 0.95
    assert probe["roc_auc_mean"] > 0.99
