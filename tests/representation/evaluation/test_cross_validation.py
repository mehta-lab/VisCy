"""Tests for leave-one-dataset-out cross-validation (cross_validation.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from applications.DynaCLR.evaluation.linear_classifiers.cross_validation import (
    _check_class_safety,
    _compute_summary,
    cross_validate_datasets,
    generate_cv_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cv_config(eval_config):
    """Eval config with wandb disabled for CV tests."""
    eval_config.wandb_logging = False
    return eval_config


@pytest.fixture
def cv_results(cv_config):
    """Run CV once and return (results_df, summary_df)."""
    return cross_validate_datasets(
        cv_config, ranking_metric="auroc", n_bootstrap=2, min_class_samples=5
    )


# ---------------------------------------------------------------------------
# Row count
# ---------------------------------------------------------------------------


class TestCVRunCount:
    def test_cv_run_count(self, cv_results):
        results_df, _ = cv_results
        # 1 model, 2 tasks, 2 channels = 4 combos
        # Each combo: 2 train datasets → baseline + 2 folds = 3 folds
        # Each fold × 2 seeds = 6 rows per combo
        # Total = 4 × 6 = 24
        n_combos = 4
        n_folds_per_combo = 3  # baseline + 2 leave-one-out
        n_seeds = 2
        expected = n_combos * n_folds_per_combo * n_seeds
        assert len(results_df) == expected, (
            f"Expected {expected} rows, got {len(results_df)}"
        )

    def test_baseline_present(self, cv_results):
        results_df, _ = cv_results
        baseline_rows = results_df[results_df["excluded_dataset"] == "baseline"]
        assert len(baseline_rows) > 0

    def test_each_combo_has_baseline(self, cv_results):
        results_df, _ = cv_results
        for (model, task, channel), group in results_df.groupby(
            ["model", "task", "channel"]
        ):
            baseline = group[group["excluded_dataset"] == "baseline"]
            assert len(baseline) > 0, f"No baseline for {model}/{task}/{channel}"


# ---------------------------------------------------------------------------
# Delta and impact
# ---------------------------------------------------------------------------


class TestDeltaComputation:
    def test_delta_computation(self, cv_results):
        _, summary_df = cv_results
        for _, row in summary_df.iterrows():
            if row["impact"] == "baseline":
                assert row["delta"] == 0.0
            elif row["impact"] != "unsafe":
                expected_delta = row["mean_auroc"] - row["baseline_mean"]
                assert np.isclose(row["delta"], expected_delta, atol=1e-10), (
                    f"Delta mismatch for {row['excluded_dataset']}: "
                    f"{row['delta']} != {expected_delta}"
                )

    def test_impact_labels_valid(self, cv_results):
        _, summary_df = cv_results
        valid_impacts = {"helps", "hurts", "uncertain", "baseline", "unsafe"}
        for impact in summary_df["impact"].values:
            assert impact in valid_impacts, f"Invalid impact label: {impact}"

    def test_baseline_has_baseline_impact(self, cv_results):
        _, summary_df = cv_results
        baseline = summary_df[summary_df["excluded_dataset"] == "baseline"]
        assert (baseline["impact"] == "baseline").all()


# ---------------------------------------------------------------------------
# Unsafe detection
# ---------------------------------------------------------------------------


class TestUnsafeDetection:
    def test_unsafe_detection(self, cv_config, tmp_path_factory):
        """Fold with too few class samples is marked unsafe."""
        results_df, summary_df = cross_validate_datasets(
            cv_config,
            ranking_metric="auroc",
            n_bootstrap=2,
            # Set threshold very high so all leave-one-out folds fail
            min_class_samples=10000,
        )
        non_baseline = summary_df[summary_df["excluded_dataset"] != "baseline"]
        assert (non_baseline["impact"] == "unsafe").all(), (
            "Expected all leave-one-out folds to be unsafe with high threshold"
        )

    def test_check_class_safety_pass(self, cv_config):
        """Safety check passes with low threshold."""
        from applications.DynaCLR.evaluation.linear_classifiers.cross_validation import (
            _build_datasets_for_combo,
        )

        model_spec = list(cv_config.models.values())[0]
        pairs = _build_datasets_for_combo(
            model_spec.train_datasets, "phase", "infection_state"
        )
        ds_dicts = [p[1] for p in pairs]
        assert _check_class_safety(ds_dicts, "infection_state", 1)

    def test_check_class_safety_fail(self, cv_config):
        """Safety check fails with very high threshold."""
        from applications.DynaCLR.evaluation.linear_classifiers.cross_validation import (
            _build_datasets_for_combo,
        )

        model_spec = list(cv_config.models.values())[0]
        pairs = _build_datasets_for_combo(
            model_spec.train_datasets, "phase", "infection_state"
        )
        ds_dicts = [p[1] for p in pairs]
        assert not _check_class_safety(ds_dicts, "infection_state", 10000)


# ---------------------------------------------------------------------------
# Metric presence
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_auroc_computed(self, cv_results):
        results_df, _ = cv_results
        auroc_vals = results_df["auroc"].dropna()
        assert len(auroc_vals) > 0, "No AUROC values computed"
        assert (auroc_vals >= 0).all() and (auroc_vals <= 1).all(), (
            "AUROC out of [0, 1] range"
        )

    def test_minority_metrics(self, cv_results):
        results_df, _ = cv_results
        baseline = results_df[results_df["excluded_dataset"] == "baseline"]
        has_minority = baseline["minority_f1"].notna().any()
        if has_minority:
            assert "minority_recall" in results_df.columns
            assert "minority_precision" in results_df.columns

    def test_annotation_counts(self, cv_results):
        results_df, _ = cv_results
        class_cols = [c for c in results_df.columns if c.startswith("train_class_")]
        assert len(class_cols) > 0, "No train class count columns found"

    def test_test_accuracy_present(self, cv_results):
        results_df, _ = cv_results
        baseline = results_df[results_df["excluded_dataset"] == "baseline"]
        assert "test_accuracy" in baseline.columns
        assert baseline["test_accuracy"].notna().any()


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


class TestCSVOutput:
    def test_csv_output(self, cv_config, cv_results):
        results_path = cv_config.output_dir / "cv_results.csv"
        summary_path = cv_config.output_dir / "cv_summary.csv"
        assert results_path.exists(), "cv_results.csv not written"
        assert summary_path.exists(), "cv_summary.csv not written"

        results = pd.read_csv(results_path)
        summary = pd.read_csv(summary_path)
        assert len(results) > 0
        assert len(summary) > 0
        assert "excluded_dataset" in results.columns
        assert "impact" in summary.columns


# ---------------------------------------------------------------------------
# No WandB calls
# ---------------------------------------------------------------------------


class TestNoWandB:
    def test_no_wandb_calls(self, cv_config, monkeypatch):
        """WandB should not be called when wandb_logging=False."""
        wandb_called = {"init": False}

        def mock_init(*args, **kwargs):
            wandb_called["init"] = True
            raise RuntimeError("wandb.init should not be called")

        monkeypatch.setattr(
            "viscy.representation.evaluation.linear_classifier.wandb.init",
            mock_init,
        )

        cv_config.wandb_logging = False
        # This should not trigger wandb since CV doesn't call save_pipeline_to_wandb
        results_df, _ = cross_validate_datasets(
            cv_config, n_bootstrap=1, min_class_samples=5
        )
        assert not wandb_called["init"], "wandb.init was called during CV"


# ---------------------------------------------------------------------------
# Single dataset edge case
# ---------------------------------------------------------------------------


class TestSingleDataset:
    def test_single_dataset_skips(self, cv_config):
        """With only 1 training dataset, leave-one-out is not possible."""
        model_spec = list(cv_config.models.values())[0]
        model_spec.train_datasets = [model_spec.train_datasets[0]]

        results_df, summary_df = cross_validate_datasets(
            cv_config, n_bootstrap=2, min_class_samples=5
        )
        # Should produce empty results since we can't do leave-one-out with 1 dataset
        assert results_df.empty


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------


class TestPDFReport:
    def test_report_generated(self, cv_config, cv_results):
        results_df, summary_df = cv_results
        report_path = generate_cv_report(
            cv_config, results_df, summary_df, ranking_metric="auroc"
        )
        assert report_path.exists()
        assert report_path.suffix == ".pdf"


# ---------------------------------------------------------------------------
# _compute_summary unit test
# ---------------------------------------------------------------------------


class TestComputeSummary:
    def test_empty_input(self):
        result = _compute_summary(pd.DataFrame(), "auroc")
        assert result.empty

    def test_correct_columns(self, cv_results):
        _, summary_df = cv_results
        expected_cols = {
            "model",
            "task",
            "channel",
            "excluded_dataset",
            "mean_auroc",
            "std_auroc",
            "baseline_mean",
            "delta",
            "impact",
        }
        assert expected_cols.issubset(set(summary_df.columns))
