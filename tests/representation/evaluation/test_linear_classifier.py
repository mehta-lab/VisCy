import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from pydantic import ValidationError

from viscy.representation.evaluation.linear_classifier import (
    LinearClassifierPipeline,
    load_and_combine_datasets,
    predict_with_classifier,
    train_linear_classifier,
)
from viscy.representation.evaluation.linear_classifier_config import (
    LinearClassifierInferenceConfig,
    LinearClassifierTrainConfig,
)


@pytest.fixture
def synthetic_adata_with_unknowns():
    """AnnData with 'unknown' and NaN labels mixed into cell_death_state."""
    rng = np.random.default_rng(42)
    n_samples = 30
    X = rng.standard_normal((n_samples, 16)).astype(np.float32)

    labels = (
        ["alive"] * 8
        + ["dead"] * 8
        + ["apoptotic"] * 8
        + ["unknown"] * 3
        + [np.nan] * 3
    )

    obs = pd.DataFrame(
        {
            "fov_name": [f"A/{(i % 4) + 1}/0" for i in range(n_samples)],
            "id": np.arange(n_samples),
            "cell_death_state": labels,
        }
    )

    return ad.AnnData(X=X, obs=obs)


class TestLinearClassifierPipeline:
    """Tests for the LinearClassifierPipeline class."""

    @pytest.fixture
    def trained_pipeline(self, annotated_adata):
        """Train a simple pipeline for reuse in tests."""
        pipeline, _ = train_linear_classifier(
            annotated_adata, task="cell_death_state", use_scaling=True, use_pca=False
        )
        return pipeline

    def test_transform_with_scaler_and_pca(self, annotated_adata):
        pipeline, _ = train_linear_classifier(
            annotated_adata,
            task="cell_death_state",
            use_scaling=True,
            use_pca=True,
            n_pca_components=5,
        )
        X = annotated_adata.X
        X_transformed = pipeline.transform(X)
        assert X_transformed.shape == (X.shape[0], 5)

    def test_transform_scaler_only(self, annotated_adata):
        pipeline, _ = train_linear_classifier(
            annotated_adata,
            task="cell_death_state",
            use_scaling=True,
            use_pca=False,
        )
        X = annotated_adata.X
        X_transformed = pipeline.transform(X)
        assert X_transformed.shape == X.shape
        assert pipeline.pca is None

    def test_transform_no_preprocessing(self, annotated_adata):
        pipeline, _ = train_linear_classifier(
            annotated_adata,
            task="cell_death_state",
            use_scaling=False,
            use_pca=False,
        )
        X = annotated_adata.X.copy()
        X_transformed = pipeline.transform(X)
        np.testing.assert_array_equal(X_transformed, X)

    def test_predict_returns_labels(self, trained_pipeline, annotated_adata):
        predictions = trained_pipeline.predict(annotated_adata.X)
        assert predictions.shape == (annotated_adata.n_obs,)
        assert set(predictions).issubset({"alive", "dead", "apoptotic"})

    def test_predict_proba_shape(self, trained_pipeline, annotated_adata):
        proba = trained_pipeline.predict_proba(annotated_adata.X)
        n_classes = len(trained_pipeline.classifier.classes_)
        assert proba.shape == (annotated_adata.n_obs, n_classes)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestTrainLinearClassifier:
    """Tests for the train_linear_classifier function."""

    def test_train_basic(self, annotated_adata):
        pipeline, metrics = train_linear_classifier(
            annotated_adata, task="cell_death_state"
        )
        assert isinstance(pipeline, LinearClassifierPipeline)
        assert isinstance(metrics, dict)
        assert "train_accuracy" in metrics
        assert "train_weighted_f1" in metrics

    def test_train_with_scaling(self, annotated_adata):
        pipeline, _ = train_linear_classifier(
            annotated_adata, task="cell_death_state", use_scaling=True
        )
        assert pipeline.scaler is not None

    def test_train_with_pca(self, annotated_adata):
        pipeline, _ = train_linear_classifier(
            annotated_adata,
            task="cell_death_state",
            use_pca=True,
            n_pca_components=5,
        )
        assert pipeline.pca is not None
        assert pipeline.pca.n_components == 5

    def test_train_no_split(self, annotated_adata):
        pipeline, metrics = train_linear_classifier(
            annotated_adata, task="cell_death_state", split_train_data=1.0
        )
        assert "train_accuracy" in metrics
        assert "val_accuracy" not in metrics

    def test_train_metrics_keys(self, annotated_adata):
        _, metrics = train_linear_classifier(
            annotated_adata, task="cell_death_state", split_train_data=0.8
        )
        assert "train_accuracy" in metrics
        assert "train_weighted_f1" in metrics
        for class_name in ["alive", "dead", "apoptotic"]:
            assert f"train_{class_name}_f1" in metrics

    def test_train_reproducibility(self, annotated_adata):
        _, metrics_a = train_linear_classifier(
            annotated_adata, task="cell_death_state", random_seed=123
        )
        _, metrics_b = train_linear_classifier(
            annotated_adata, task="cell_death_state", random_seed=123
        )
        assert metrics_a == metrics_b

    def test_train_sparse_matrix(self, annotated_adata):
        sparse_adata = annotated_adata.copy()
        sparse_adata.X = scipy.sparse.csr_matrix(sparse_adata.X)
        pipeline, metrics = train_linear_classifier(
            sparse_adata, task="cell_death_state"
        )
        assert isinstance(pipeline, LinearClassifierPipeline)
        assert "train_accuracy" in metrics


class TestPredictWithClassifier:
    """Tests for the predict_with_classifier function."""

    @pytest.fixture
    def pipeline_and_adata(self, annotated_adata):
        pipeline, _ = train_linear_classifier(annotated_adata, task="cell_death_state")
        return pipeline, annotated_adata

    def test_predict_adds_obs_columns(self, pipeline_and_adata):
        pipeline, adata = pipeline_and_adata
        result = predict_with_classifier(adata.copy(), pipeline, "cell_death_state")
        assert "predicted_cell_death_state" in result.obs.columns

    def test_predict_adds_obsm_proba(self, pipeline_and_adata):
        pipeline, adata = pipeline_and_adata
        result = predict_with_classifier(adata.copy(), pipeline, "cell_death_state")
        assert "predicted_cell_death_state_proba" in result.obsm
        n_classes = len(pipeline.classifier.classes_)
        assert result.obsm["predicted_cell_death_state_proba"].shape == (
            adata.n_obs,
            n_classes,
        )

    def test_predict_adds_uns_classes(self, pipeline_and_adata):
        pipeline, adata = pipeline_and_adata
        result = predict_with_classifier(adata.copy(), pipeline, "cell_death_state")
        assert "predicted_cell_death_state_classes" in result.uns
        assert result.uns["predicted_cell_death_state_classes"] == list(
            pipeline.classifier.classes_
        )


class TestLoadAndCombineDatasets:
    """Tests for the load_and_combine_datasets function."""

    def test_single_dataset(self, annotated_adata_zarr):
        combined = load_and_combine_datasets(
            [annotated_adata_zarr], task="cell_death_state"
        )
        assert isinstance(combined, ad.AnnData)
        assert combined.n_obs > 0

    def test_filters_unknown_labels(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 20
        X = rng.standard_normal((n, 8)).astype(np.float32)
        labels = ["alive"] * 10 + ["unknown"] * 10
        obs = pd.DataFrame({"fov_name": ["A/1/0"] * n, "id": range(n)})
        adata = ad.AnnData(X=X, obs=obs)

        zarr_path = tmp_path / "emb.zarr"
        adata.write_zarr(zarr_path)

        csv_path = tmp_path / "ann.csv"
        ann_df = pd.DataFrame(
            {"fov_name": ["A/1/0"] * n, "id": range(n), "cell_death_state": labels}
        )
        ann_df.to_csv(csv_path, index=False)

        combined = load_and_combine_datasets(
            [{"embeddings": zarr_path, "annotations": csv_path}],
            task="cell_death_state",
        )
        assert "unknown" not in combined.obs["cell_death_state"].values

    def test_filters_nan_labels(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 20
        X = rng.standard_normal((n, 8)).astype(np.float32)
        labels = ["alive"] * 10 + [np.nan] * 10
        obs = pd.DataFrame({"fov_name": ["A/1/0"] * n, "id": range(n)})
        adata = ad.AnnData(X=X, obs=obs)

        zarr_path = tmp_path / "emb.zarr"
        adata.write_zarr(zarr_path)

        csv_path = tmp_path / "ann.csv"
        ann_df = pd.DataFrame(
            {"fov_name": ["A/1/0"] * n, "id": range(n), "cell_death_state": labels}
        )
        ann_df.to_csv(csv_path, index=False)

        combined = load_and_combine_datasets(
            [{"embeddings": zarr_path, "annotations": csv_path}],
            task="cell_death_state",
        )
        assert combined.obs["cell_death_state"].notna().all()

    def test_raises_on_empty(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 10
        X = rng.standard_normal((n, 8)).astype(np.float32)
        obs = pd.DataFrame({"fov_name": ["A/1/0"] * n, "id": range(n)})
        adata = ad.AnnData(X=X, obs=obs)

        zarr_path = tmp_path / "emb.zarr"
        adata.write_zarr(zarr_path)

        csv_path = tmp_path / "ann.csv"
        ann_df = pd.DataFrame(
            {
                "fov_name": ["A/1/0"] * n,
                "id": range(n),
                "cell_death_state": ["unknown"] * n,
            }
        )
        ann_df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="No training data loaded"):
            load_and_combine_datasets(
                [{"embeddings": zarr_path, "annotations": csv_path}],
                task="cell_death_state",
            )

    def test_multiple_datasets(self, annotated_adata_zarr, tmp_path):
        rng = np.random.default_rng(99)
        n = 30
        X = rng.standard_normal((n, 16)).astype(np.float32)
        labels = ["alive"] * 15 + ["dead"] * 15
        obs = pd.DataFrame({"fov_name": ["B/1/0"] * n, "id": range(n)})
        adata = ad.AnnData(X=X, obs=obs)

        zarr_path = tmp_path / "emb2.zarr"
        adata.write_zarr(zarr_path)

        csv_path = tmp_path / "ann2.csv"
        ann_df = pd.DataFrame(
            {"fov_name": ["B/1/0"] * n, "id": range(n), "cell_death_state": labels}
        )
        ann_df.to_csv(csv_path, index=False)

        dataset2 = {"embeddings": zarr_path, "annotations": csv_path}
        combined = load_and_combine_datasets(
            [annotated_adata_zarr, dataset2], task="cell_death_state"
        )
        assert combined.n_obs == 90


class TestLinearClassifierTrainConfig:
    """Tests for the LinearClassifierTrainConfig pydantic model."""

    def _make_dataset(self, tmp_path, suffix=""):
        zarr_path = tmp_path / f"emb{suffix}.zarr"
        zarr_path.mkdir()
        csv_path = tmp_path / f"ann{suffix}.csv"
        csv_path.write_text("fov_name,id,cell_death_state\nA/1/0,0,alive\n")
        return {"embeddings": str(zarr_path), "annotations": str(csv_path)}

    def test_valid_config(self, tmp_path):
        dataset = self._make_dataset(tmp_path)
        config = LinearClassifierTrainConfig(
            task="cell_death_state",
            input_channel="phase",
            embedding_model="test_model",
            train_datasets=[dataset],
            wandb_project="test_project",
        )
        assert config.task == "cell_death_state"

    def test_invalid_task(self, tmp_path):
        dataset = self._make_dataset(tmp_path)
        with pytest.raises(ValidationError):
            LinearClassifierTrainConfig(
                task="invalid_task",
                input_channel="phase",
                embedding_model="test_model",
                train_datasets=[dataset],
                wandb_project="test_project",
            )

    def test_invalid_channel(self, tmp_path):
        dataset = self._make_dataset(tmp_path)
        with pytest.raises(ValidationError):
            LinearClassifierTrainConfig(
                task="cell_death_state",
                input_channel="invalid_channel",
                embedding_model="test_model",
                train_datasets=[dataset],
                wandb_project="test_project",
            )

    def test_pca_without_components(self, tmp_path):
        dataset = self._make_dataset(tmp_path)
        with pytest.raises(ValidationError, match="n_pca_components"):
            LinearClassifierTrainConfig(
                task="cell_death_state",
                input_channel="phase",
                embedding_model="test_model",
                train_datasets=[dataset],
                use_pca=True,
                n_pca_components=None,
                wandb_project="test_project",
            )

    def test_missing_dataset_keys(self, tmp_path):
        with pytest.raises(ValidationError, match="embeddings"):
            LinearClassifierTrainConfig(
                task="cell_death_state",
                input_channel="phase",
                embedding_model="test_model",
                train_datasets=[{"only_embeddings": "/some/path"}],
                wandb_project="test_project",
            )

    def test_nonexistent_paths(self, tmp_path):
        with pytest.raises(ValidationError, match="not found"):
            LinearClassifierTrainConfig(
                task="cell_death_state",
                input_channel="phase",
                embedding_model="test_model",
                train_datasets=[
                    {
                        "embeddings": "/nonexistent/path.zarr",
                        "annotations": "/nonexistent/ann.csv",
                    }
                ],
                wandb_project="test_project",
            )


class TestLinearClassifierInferenceConfig:
    """Tests for the LinearClassifierInferenceConfig pydantic model."""

    def test_valid_config(self, tmp_path):
        emb = tmp_path / "emb.zarr"
        emb.mkdir()
        config = LinearClassifierInferenceConfig(
            wandb_project="test_project",
            model_name="test_model",
            embeddings_path=str(emb),
            output_path=str(tmp_path / "output.zarr"),
        )
        assert config.embeddings_path == str(emb)

    def test_missing_embeddings(self, tmp_path):
        with pytest.raises(ValidationError, match="not found"):
            LinearClassifierInferenceConfig(
                wandb_project="test_project",
                model_name="test_model",
                embeddings_path=str(tmp_path / "nonexistent.zarr"),
                output_path=str(tmp_path / "output.zarr"),
            )

    def test_output_exists_no_overwrite(self, tmp_path):
        emb = tmp_path / "emb.zarr"
        emb.mkdir()
        out = tmp_path / "output.zarr"
        out.mkdir()
        with pytest.raises(ValidationError, match="already exists"):
            LinearClassifierInferenceConfig(
                wandb_project="test_project",
                model_name="test_model",
                embeddings_path=str(emb),
                output_path=str(out),
                overwrite=False,
            )

    def test_output_exists_with_overwrite(self, tmp_path):
        emb = tmp_path / "emb.zarr"
        emb.mkdir()
        out = tmp_path / "output.zarr"
        out.mkdir()
        config = LinearClassifierInferenceConfig(
            wandb_project="test_project",
            model_name="test_model",
            embeddings_path=str(emb),
            output_path=str(out),
            overwrite=True,
        )
        assert config.overwrite is True
