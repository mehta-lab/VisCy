"""Tests for organelle remodeling support in linear classifier.

Covers: marker-namespaced tasks, well filtering, artifact provenance,
optional output_path, and include_wells config fields.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from viscy_utils.evaluation.linear_classifier import (
    predict_with_classifier,
    train_linear_classifier,
)
from viscy_utils.evaluation.linear_classifier_config import (
    LinearClassifierInferenceConfig,
)


@pytest.fixture
def annotated_adata() -> ad.AnnData:
    rng = np.random.default_rng(42)
    n_samples = 60
    n_features = 16
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    fov_names = [f"A/{(i % 4) + 1}/0" for i in range(n_samples)]
    labels = (["alive"] * 20) + (["dead"] * 20) + (["apoptotic"] * 20)
    obs = pd.DataFrame(
        {
            "fov_name": fov_names,
            "id": np.arange(n_samples),
            "cell_death_state": labels,
        }
    )
    return ad.AnnData(X=X, obs=obs)


@pytest.fixture
def pipeline_and_adata(annotated_adata):
    pipeline, _ = train_linear_classifier(annotated_adata, task="cell_death_state")
    return pipeline, annotated_adata


class TestPredictOrganelle:
    def test_predict_stores_provenance(self, pipeline_and_adata):
        pipeline, adata = pipeline_and_adata
        metadata = {
            "artifact_name": "linear-classifier-cell_death_state-phase:v2",
            "artifact_id": "abc123",
            "artifact_version": "v2",
        }
        result = predict_with_classifier(adata.copy(), pipeline, "cell_death_state", artifact_metadata=metadata)
        assert result.uns["classifier_cell_death_state_artifact"] == "linear-classifier-cell_death_state-phase:v2"
        assert result.uns["classifier_cell_death_state_id"] == "abc123"
        assert result.uns["classifier_cell_death_state_version"] == "v2"

    def test_predict_no_provenance_by_default(self, pipeline_and_adata):
        pipeline, adata = pipeline_and_adata
        result = predict_with_classifier(adata.copy(), pipeline, "cell_death_state")
        assert "classifier_cell_death_state_artifact" not in result.uns
        assert "classifier_cell_death_state_id" not in result.uns
        assert "classifier_cell_death_state_version" not in result.uns

    def test_predict_with_include_wells(self, pipeline_and_adata):
        pipeline, adata = pipeline_and_adata
        data = adata.copy()
        result = predict_with_classifier(data, pipeline, "cell_death_state", include_wells=["A/1"])
        well_mask = result.obs["fov_name"].str.startswith("A/1/")
        predicted = result.obs["predicted_cell_death_state"]
        assert predicted[well_mask].notna().all()
        assert predicted[~well_mask].isna().all()

        proba = result.obsm["predicted_cell_death_state_proba"]
        assert np.isfinite(proba[well_mask]).all()
        assert np.isnan(proba[~well_mask]).all()

    def test_predict_marker_namespaced_task(self, pipeline_and_adata):
        pipeline, adata = pipeline_and_adata
        result = predict_with_classifier(
            adata.copy(),
            pipeline,
            "organelle_state_g3bp1",
            include_wells=["A/1"],
        )
        assert "predicted_organelle_state_g3bp1" in result.obs.columns
        assert "predicted_organelle_state_g3bp1_proba" in result.obsm
        assert "predicted_organelle_state_g3bp1_classes" in result.uns


class TestLinearClassifierInferenceConfigOrganelle:
    def test_output_path_none_defaults_to_inplace(self, tmp_path):
        emb = tmp_path / "emb.zarr"
        emb.mkdir()
        config = LinearClassifierInferenceConfig(
            wandb_project="test_project",
            model_name="test_model",
            embeddings_path=str(emb),
        )
        assert config.output_path is None

    def test_include_wells(self, tmp_path):
        emb = tmp_path / "emb.zarr"
        emb.mkdir()
        config = LinearClassifierInferenceConfig(
            wandb_project="test_project",
            model_name="test_model",
            embeddings_path=str(emb),
            include_wells=["A/1", "B/2"],
        )
        assert config.include_wells == ["A/1", "B/2"]

    def test_include_wells_none_by_default(self, tmp_path):
        emb = tmp_path / "emb.zarr"
        emb.mkdir()
        config = LinearClassifierInferenceConfig(
            wandb_project="test_project",
            model_name="test_model",
            embeddings_path=str(emb),
        )
        assert config.include_wells is None
