"""Tests for the evaluation pipeline (evaluate_dataset.py)."""

from unittest.mock import MagicMock, patch

import joblib
import pandas as pd
import pytest
from applications.DynaCLR.evaluation.linear_classifiers.evaluate_dataset import (
    DatasetEvalConfig,
    ModelSpec,
    TrainDataset,
    infer_classifiers,
    train_classifiers,
)
from pydantic import ValidationError

from viscy.representation.evaluation.linear_classifier import (
    LinearClassifierPipeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_wandb():
    """Return a patch context that mocks all wandb calls."""
    mock_run = MagicMock()
    mock_run.summary = {}
    mock_artifact = MagicMock()
    mock_artifact.version = "v0"
    mock_run.log_artifact.return_value = mock_artifact

    return patch.multiple(
        "viscy.representation.evaluation.linear_classifier",
        wandb=MagicMock(
            init=MagicMock(return_value=mock_run),
            Artifact=MagicMock(return_value=mock_artifact),
        ),
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestDatasetEvalConfig:
    def test_valid_config(self, eval_config):
        assert eval_config.dataset_name == "test_dataset"
        assert len(eval_config.models) == 1
        assert "test_model" in eval_config.models

    def test_missing_test_annotations(self, synthetic_train_data, tmp_path):
        emb_dir = tmp_path / "fake_emb"
        emb_dir.mkdir()
        with pytest.raises(ValidationError, match="Test annotations CSV not found"):
            DatasetEvalConfig(
                dataset_name="test",
                test_annotations_csv=tmp_path / "nonexistent.csv",
                models={
                    "m": ModelSpec(
                        name="M",
                        train_datasets=synthetic_train_data,
                        test_embeddings_dir=emb_dir,
                        version="v1",
                        wandb_project="p",
                    )
                },
                output_dir=tmp_path / "out",
            )

    def test_missing_embeddings_dir(self, synthetic_train_data, tmp_path):
        csv = tmp_path / "ann.csv"
        csv.write_text("fov_name,id,infection_state\nA/1/0,0,infected\n")
        with pytest.raises(
            ValidationError, match="Test embeddings directory not found"
        ):
            DatasetEvalConfig(
                dataset_name="test",
                test_annotations_csv=csv,
                models={
                    "m": ModelSpec(
                        name="M",
                        train_datasets=synthetic_train_data,
                        test_embeddings_dir=tmp_path / "nonexistent",
                        version="v1",
                        wandb_project="p",
                    )
                },
                output_dir=tmp_path / "out",
            )

    def test_missing_train_annotations(self, tmp_path):
        emb_dir = tmp_path / "emb"
        emb_dir.mkdir()
        csv = tmp_path / "ann.csv"
        csv.write_text("fov_name,id,infection_state\nA/1/0,0,infected\n")
        with pytest.raises(ValidationError, match="Annotations file not found"):
            DatasetEvalConfig(
                dataset_name="test",
                test_annotations_csv=csv,
                models={
                    "m": ModelSpec(
                        name="M",
                        train_datasets=[
                            TrainDataset(
                                embeddings_dir=emb_dir,
                                annotations=tmp_path / "missing.csv",
                            )
                        ],
                        test_embeddings_dir=emb_dir,
                        version="v1",
                        wandb_project="p",
                    )
                },
                output_dir=tmp_path / "out",
            )

    def test_invalid_task(self, synthetic_train_data, synthetic_test_data, tmp_path):
        test_emb_dir, test_csv = synthetic_test_data
        with pytest.raises(ValidationError, match="Invalid task"):
            DatasetEvalConfig(
                dataset_name="test",
                test_annotations_csv=test_csv,
                models={
                    "m": ModelSpec(
                        name="M",
                        train_datasets=synthetic_train_data,
                        test_embeddings_dir=test_emb_dir,
                        version="v1",
                        wandb_project="p",
                    )
                },
                output_dir=tmp_path / "out",
                tasks=["not_a_valid_task"],
            )

    def test_auto_detect_tasks(
        self, synthetic_train_data, synthetic_test_data, tmp_path
    ):
        test_emb_dir, test_csv = synthetic_test_data
        config = DatasetEvalConfig(
            dataset_name="test",
            test_annotations_csv=test_csv,
            models={
                "m": ModelSpec(
                    name="M",
                    train_datasets=synthetic_train_data,
                    test_embeddings_dir=test_emb_dir,
                    version="v1",
                    wandb_project="p",
                )
            },
            output_dir=tmp_path / "out",
            tasks=None,
        )
        assert config.tasks is None


# ---------------------------------------------------------------------------
# Train classifiers
# ---------------------------------------------------------------------------


class TestTrainClassifiers:
    def test_trains_all_combinations(self, eval_config):
        with _mock_wandb():
            results = train_classifiers(eval_config)

        model_results = results["test_model"]
        expected_combos = {
            ("infection_state", "phase"),
            ("infection_state", "organelle"),
            ("cell_division_state", "phase"),
            ("cell_division_state", "organelle"),
        }
        assert set(model_results.keys()) == expected_combos

    def test_metrics_contain_val_keys(self, eval_config):
        with _mock_wandb():
            results = train_classifiers(eval_config)

        for combo_key, result in results["test_model"].items():
            metrics = result["metrics"]
            assert "val_accuracy" in metrics, f"Missing val_accuracy for {combo_key}"
            assert "val_weighted_f1" in metrics, (
                f"Missing val_weighted_f1 for {combo_key}"
            )

    def test_pipeline_saved_to_disk(self, eval_config):
        with _mock_wandb():
            train_classifiers(eval_config)

        model_dir = eval_config.output_dir / "test_model"
        for task in eval_config.tasks:
            for channel in eval_config.channels:
                pipeline_path = model_dir / f"{task}_{channel}_pipeline.joblib"
                assert pipeline_path.exists(), f"Missing {pipeline_path.name}"
                pipeline = joblib.load(pipeline_path)
                assert isinstance(pipeline, LinearClassifierPipeline)

    def test_metrics_csv_written(self, eval_config):
        with _mock_wandb():
            train_classifiers(eval_config)

        model_csv = eval_config.output_dir / "test_model" / "metrics_summary.csv"
        assert model_csv.exists()
        df = pd.read_csv(model_csv)
        assert "task" in df.columns
        assert "channel" in df.columns
        assert "val_accuracy" in df.columns
        assert len(df) == 4  # 2 tasks x 2 channels

    def test_comparison_csv_written(self, eval_config):
        with _mock_wandb():
            train_classifiers(eval_config)

        comparison_csv = eval_config.output_dir / "metrics_comparison.csv"
        assert comparison_csv.exists()
        df = pd.read_csv(comparison_csv)
        assert "model" in df.columns

    def test_skips_missing_channel(self, eval_config):
        eval_config.channels = ["phase", "organelle", "sensor"]
        with _mock_wandb():
            results = train_classifiers(eval_config)

        model_results = results["test_model"]
        trained_channels = {ch for _, ch in model_results.keys()}
        assert "sensor" not in trained_channels
        assert "phase" in trained_channels
        assert "organelle" in trained_channels

    def test_skips_missing_task(self, eval_config):
        eval_config.tasks = [
            "infection_state",
            "cell_division_state",
            "cell_death_state",
        ]
        with _mock_wandb():
            results = train_classifiers(eval_config)

        model_results = results["test_model"]
        trained_tasks = {t for t, _ in model_results.keys()}
        assert "cell_death_state" not in trained_tasks
        assert "infection_state" in trained_tasks

    def test_wandb_config_has_provenance(self, eval_config):
        mock_run = MagicMock()
        mock_run.summary = {}
        mock_artifact = MagicMock()
        mock_artifact.version = "v0"
        mock_run.log_artifact.return_value = mock_artifact

        mock_wandb_module = MagicMock()
        mock_wandb_module.init.return_value = mock_run
        mock_wandb_module.Artifact.return_value = mock_artifact

        with patch.multiple(
            "viscy.representation.evaluation.linear_classifier",
            wandb=mock_wandb_module,
        ):
            train_classifiers(eval_config)

        calls = mock_wandb_module.init.call_args_list
        assert len(calls) > 0
        for call in calls:
            wandb_config = call.kwargs.get("config", call.args[0] if call.args else {})
            assert "embedding_model" in wandb_config
            assert "test_dataset" in wandb_config
            assert "train_dataset_names" in wandb_config
            assert wandb_config["embedding_model"] == "TestModel-v1"
            assert wandb_config["test_dataset"] == "test_dataset"


# ---------------------------------------------------------------------------
# Infer classifiers
# ---------------------------------------------------------------------------


class TestInferClassifiers:
    def test_predictions_on_all_cells(self, eval_config):
        with _mock_wandb():
            trained = train_classifiers(eval_config)
        predictions = infer_classifiers(eval_config, trained=trained)

        for model_label, model_preds in predictions.items():
            for (task, channel), adata in model_preds.items():
                pred_col = f"predicted_{task}"
                assert pred_col in adata.obs.columns
                assert adata.obs[pred_col].notna().all(), (
                    f"NaN predictions for {model_label}/{task}/{channel}"
                )

    def test_prediction_columns_exist(self, eval_config):
        with _mock_wandb():
            trained = train_classifiers(eval_config)
        predictions = infer_classifiers(eval_config, trained=trained)

        for model_preds in predictions.values():
            for (task, _), adata in model_preds.items():
                assert f"predicted_{task}" in adata.obs.columns
                assert f"predicted_{task}_proba" in adata.obsm
                assert f"predicted_{task}_classes" in adata.uns

    def test_predictions_zarr_saved(self, eval_config):
        with _mock_wandb():
            trained = train_classifiers(eval_config)
        infer_classifiers(eval_config, trained=trained)

        for task in eval_config.tasks:
            for channel in eval_config.channels:
                pred_path = (
                    eval_config.output_dir
                    / "test_model"
                    / f"{task}_{channel}_predictions.zarr"
                )
                assert pred_path.exists(), f"Missing {pred_path.name}"

    def test_loads_pipeline_from_disk(self, eval_config):
        with _mock_wandb():
            train_classifiers(eval_config)

        predictions = infer_classifiers(eval_config, trained=None)
        model_preds = predictions["test_model"]
        assert len(model_preds) == 4  # 2 tasks x 2 channels

    def test_provenance_in_uns(self, eval_config):
        with _mock_wandb():
            trained = train_classifiers(eval_config)
        predictions = infer_classifiers(eval_config, trained=trained)

        for model_preds in predictions.values():
            for (task, _), adata in model_preds.items():
                assert f"classifier_{task}_artifact" in adata.uns
