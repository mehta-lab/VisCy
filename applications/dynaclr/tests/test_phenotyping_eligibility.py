"""Strict eligibility, early prediction filtering, and raw-store preprocessing."""

import json
from hashlib import sha256
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from dynaclr.evaluation import predict_triplet
from dynaclr.evaluation.phenotyping_eligibility import (
    ELIGIBILITY_COLUMNS,
    EligibilityTripletMixin,
    eligibility_records_sha256,
    filter_embedding_store,
    filter_observations,
    load_eligibility,
    main,
    read_embedding_obs,
)


@pytest.fixture
def eligibility():
    frame = pd.DataFrame(
        [
            ["plate", "A/1/000", 1, 0, "GFP", False, "no organelle"],
            ["plate", "A/1/000", 1, 1, "GFP", pd.NA, "not reviewed"],
            ["plate", "A/1/000", 1, 2, "GFP", True, "reviewed"],
            ["other", "A/1/000", 1, 0, "GFP", False, "other dataset"],
        ],
        columns=ELIGIBILITY_COLUMNS,
    )
    frame["eligible"] = frame["eligible"].astype("boolean")
    return frame


@pytest.fixture
def obs():
    return pd.DataFrame(
        {"fov_name": ["A/1/000"] * 4, "track_id": [1] * 4, "t": [0, 1, 2, 3], "payload": [9, 8, 7, 6]},
        index=["repeat", "repeat", "z", "a"],
    )


def test_explicit_false_only_preserves_order_index_and_inputs(obs, eligibility):
    original_obs, original_eligibility = obs.copy(deep=True), eligibility.copy(deep=True)
    filtered, audit = filter_observations(obs, eligibility, "plate", "GFP")
    pd.testing.assert_frame_equal(filtered, obs.iloc[1:])
    pd.testing.assert_frame_equal(obs, original_obs)
    pd.testing.assert_frame_equal(eligibility, original_eligibility)
    assert {
        key: audit[key]
        for key in (
            "n_input",
            "n_retained",
            "n_excluded",
            "n_explicit_true",
            "n_unknown",
            "n_missing",
            "n_matched",
            "n_unused_eligibility_records",
        )
    } == {
        "n_input": 4,
        "n_retained": 3,
        "n_excluded": 1,
        "n_explicit_true": 1,
        "n_unknown": 1,
        "n_missing": 1,
        "n_matched": 3,
        "n_unused_eligibility_records": 1,
    }
    assert audit["excluded_reason_counts"] == {"no organelle": 1}
    json.dumps(audit)


def test_channel_decisions_do_not_drop_phase_or_other_acquisitions(obs, eligibility):
    phase, audit = filter_observations(obs, eligibility, "plate", "Phase3D")
    pd.testing.assert_frame_equal(phase, obs)
    assert audit["n_missing"] == 4
    other, audit = filter_observations(obs, eligibility, "unseen", "GFP")
    pd.testing.assert_frame_equal(other, obs)
    assert audit["n_missing"] == 4
    mixed = pd.concat(
        [obs.assign(dataset="plate", channel_name="GFP"), obs.assign(dataset="plate", channel_name="Phase3D")]
    )
    filtered, audit = filter_observations(mixed, eligibility)
    assert len(filtered) == 7
    assert filtered["channel_name"].eq("Phase3D").sum() == 4


def test_all_unknown_and_empty_tables_keep_every_observation(obs, eligibility):
    eligibility["eligible"] = pd.NA
    filtered, audit = filter_observations(obs, eligibility, "plate", "GFP")
    pd.testing.assert_frame_equal(filtered, obs)
    assert audit["n_unknown"] == 3
    filtered, audit = filter_observations(obs, eligibility.iloc[:0], "plate", "GFP")
    pd.testing.assert_frame_equal(filtered, obs)
    assert audit["n_missing"] == 4
    empty, audit = filter_observations(obs.iloc[:0], eligibility, "plate", "GFP")
    assert empty.empty and audit["n_input"] == 0


@pytest.mark.parametrize("source", ["obs", "eligibility"])
def test_duplicate_keys_are_rejected_before_scoping(obs, eligibility, source):
    if source == "obs":
        obs = pd.concat([obs, obs.iloc[:1]])
    else:
        # Even a duplicate outside the requested dataset must be rejected.
        eligibility = pd.concat([eligibility, eligibility.iloc[-1:]])
    with pytest.raises(ValueError, match="duplicate observation keys"):
        filter_observations(obs, eligibility, "plate", "GFP")


@pytest.mark.parametrize(
    "column,value", [("track_id", "1"), ("t", 0.0), ("track_id", True), ("fov_name", ""), ("fov_name", None)]
)
def test_invalid_observation_keys_fail(obs, eligibility, column, value):
    obs[column] = obs[column].astype(object)
    obs.iloc[0, obs.columns.get_loc(column)] = value
    with pytest.raises(ValueError, match="key"):
        filter_observations(obs, eligibility, "plate", "GFP")


@pytest.mark.parametrize("value", [0, 1, "False", "True"])
def test_non_boolean_eligibility_fails(obs, eligibility, value):
    eligibility["eligible"] = eligibility["eligible"].astype(object)
    eligibility.iloc[0, eligibility.columns.get_loc("eligible")] = value
    with pytest.raises(ValueError, match="booleans or null"):
        filter_observations(obs, eligibility, "plate", "GFP")


def test_context_is_required_and_cannot_override_existing_values(obs, eligibility):
    with pytest.raises(ValueError, match="missing key columns"):
        filter_observations(obs, eligibility)
    with pytest.raises(ValueError, match="conflicts"):
        filter_observations(obs.assign(channel_name="Phase3D"), eligibility, "plate", "GFP")
    with pytest.raises(ValueError, match="conflicts"):
        filter_observations(obs.assign(dataset="different"), eligibility, "plate", "GFP")
    with pytest.raises(ValueError, match="missing columns"):
        filter_observations(obs, eligibility.drop(columns="reason"), "plate", "GFP")


def test_hashes_are_canonical_but_file_sha_is_authoritative(eligibility, tmp_path):
    digest = eligibility_records_sha256(eligibility)
    reordered = eligibility.iloc[::-1].reset_index(drop=True)
    assert eligibility_records_sha256(reordered) == digest
    changed = eligibility.copy()
    changed.loc[0, "eligible"] = True
    assert eligibility_records_sha256(changed) != digest
    changed = eligibility.copy()
    changed.loc[0, "reason"] = "different review"
    assert eligibility_records_sha256(changed) != digest
    path = tmp_path / "eligibility.parquet"
    eligibility.to_parquet(path)
    table, provenance = load_eligibility(path)
    assert provenance["eligibility_file_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert provenance["eligibility_records_sha256"] == digest
    pd.testing.assert_frame_equal(table, eligibility)


def test_fov_tables_are_filtered_before_patch_access(obs, eligibility):
    class MetadataOnlyBase:
        def __init__(self):
            self.positions = [
                SimpleNamespace(zgroup=SimpleNamespace(name="/A/1/000")),
                SimpleNamespace(zgroup=SimpleNamespace(name="/B/1/000")),
            ]
            self.tables = [obs.reset_index(drop=True), obs.iloc[:1].reset_index(drop=True)]

        def _align_tracks_tables_with_positions(self):
            return self.positions, self.tables

    class EligibleDataModule(EligibilityTripletMixin, MetadataOnlyBase):
        pass

    audit = {}
    module = EligibleDataModule(
        eligibility_table=eligibility, eligibility_dataset="plate", eligibility_channel="GFP", eligibility_audit=audit
    )
    positions, tables = module._align_tracks_tables_with_positions()
    assert positions == module.positions
    pd.testing.assert_frame_equal(tables[0], module.tables[0].iloc[1:])
    pd.testing.assert_frame_equal(tables[1], module.tables[1])
    assert audit["n_input"] == 5 and audit["n_missing"] == 2
    # The first FOV becomes empty; the second remains eligible by default.
    module._eligibility_aligned_tables = None
    module.tables = [obs.iloc[:1], obs.iloc[:1]]
    positions, tables = module._align_tracks_tables_with_positions()
    assert positions == module.positions[1:]
    assert len(tables) == 1 and not tables[0].empty
    assert module._align_tracks_tables_with_positions() == (positions, tables)
    module._eligibility_aligned_tables = None
    module.tables = [obs.iloc[:1], obs.iloc[:0]]
    with pytest.raises(ValueError, match="excluded all.*no image patches"):
        module._align_tracks_tables_with_positions()


@pytest.mark.parametrize("suffix", [".h5ad", ".zarr"])
def test_new_store_subsets_raw_slots_and_removes_fitted_representations(obs, eligibility, tmp_path, suffix):
    path, output = tmp_path / f"source{suffix}", tmp_path / f"filtered{suffix}"
    parquet = tmp_path / "eligibility.parquet"
    eligibility.to_parquet(parquet)
    # Object strings avoid the pandas 3 / AnnData 0.12 Arrow string limitation.
    obs = obs.copy()
    obs["fov_name"] = obs["fov_name"].astype(object)
    obs.index = pd.Index(["a", "b", "c", "d"], dtype=object)
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    source = ad.AnnData(X=values, obs=obs, var=pd.DataFrame(index=pd.Index(["x", "y", "z"], dtype=object)))
    source.obsm["X_backbone"] = values.copy()
    source.obsm["X_projections"] = values[:, :2].copy()
    source.obsm["X_normalized_pca80"] = values[:, :2].copy()
    source.obsm["X_umap"] = values[:, :2].copy()
    source.obsp["connectivities"] = np.eye(4)
    source.layers["raw_values"] = values.copy()
    source.uns["model_family"] = "test"
    source.uns["X_normalized_pca80"] = {"fit_cells": 4}
    source.write_h5ad(path) if suffix == ".h5ad" else source.write_zarr(path)
    pd.testing.assert_frame_equal(read_embedding_obs(path), source.obs)
    audit = filter_embedding_store(path, output, parquet, dataset="plate", channel_name="GFP")
    result = ad.read_h5ad(output) if suffix == ".h5ad" else ad.read_zarr(output)
    np.testing.assert_array_equal(result.X, values[1:])
    np.testing.assert_array_equal(result.layers["raw_values"], values[1:])
    assert set(result.obsm) == {"X_backbone", "X_projections"}
    assert not result.obsp
    assert set(result.uns) == {"phenotyping_eligibility"}
    assert json.loads(result.uns["phenotyping_eligibility"])["requires_representation_refit"] is True
    sidecar = json.loads((tmp_path / f"filtered{suffix}.eligibility.json").read_text())
    assert sidecar["source_uns"]["model_family"] == "test"
    assert audit["eligibility_file_sha256"] == sha256(parquet.read_bytes()).hexdigest()
    original = ad.read_h5ad(path) if suffix == ".h5ad" else ad.read_zarr(path)
    assert original.n_obs == 4 and "X_normalized_pca80" in original.obsm
    with pytest.raises(FileExistsError):
        filter_embedding_store(path, output, parquet, dataset="plate", channel_name="GFP")
    with pytest.raises(FileExistsError):
        filter_embedding_store(path, path, parquet, dataset="plate", channel_name="GFP")


def test_obs_only_cli_does_not_read_embeddings(obs, eligibility, tmp_path, monkeypatch):
    path = tmp_path / "eligibility.parquet"
    eligibility.to_parquet(path)
    module = __import__(main.callback.__module__, fromlist=["read_embedding_obs"])
    monkeypatch.setattr(module, "read_embedding_obs", lambda _: obs)
    monkeypatch.setattr(ad, "read_zarr", lambda *_: pytest.fail("Full store read during obs-only audit"))
    result = CliRunner().invoke(
        main, ["--input", str(path), "--eligibility", str(path), "--dataset", "plate", "--channel-name", "GFP"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["n_excluded"] == 1
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("enabled", [False, True])
def test_prediction_cli_loads_eligibility_once_and_passes_exact_context(eligibility, tmp_path, monkeypatch, enabled):
    parquet = tmp_path / "eligibility.parquet"
    eligibility.to_parquet(parquet)
    collection = tmp_path / "collection.yaml"
    collection.write_text("unused: true\n")
    entry = SimpleNamespace(name="plate")
    run = SimpleNamespace(
        experiment="plate", marker="SEC61B", channel="GFP", wells=None, output_path=tmp_path / "out.zarr"
    )
    monkeypatch.setattr(predict_triplet, "load_collection", lambda _: SimpleNamespace(experiments=[entry]))
    monkeypatch.setattr(predict_triplet, "plan_predict_runs", lambda *a, **k: [run])
    calls = []
    monkeypatch.setattr(predict_triplet, "_run_predict", lambda *a, **k: calls.append((a, k)))
    args = [
        "-c",
        str(collection),
        "--model-type",
        "foundation",
        "--training-config",
        str(collection),
        "--model-family",
        "test",
        "--run",
        "test",
        "--ckpt-name",
        "test",
        "--no-enrich-obs",
    ]
    if enabled:
        args.extend(["--eligibility", str(parquet)])
    result = CliRunner().invoke(predict_triplet.main, args)
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and calls[0][0][0] is run
    kwargs = calls[0][1]
    if enabled:
        pd.testing.assert_frame_equal(kwargs["eligibility_table"], eligibility)
        assert kwargs["eligibility_provenance"]["eligibility_file_sha256"] == sha256(parquet.read_bytes()).hexdigest()
    else:
        assert "eligibility_table" not in kwargs


def test_categorical_keys_and_review_fields_match_and_hash(obs, eligibility):
    categorical = eligibility.copy()
    for column in ("dataset", "fov_name", "channel_name", "reason"):
        categorical[column] = categorical[column].astype("category")
    categorical_obs = obs.copy()
    categorical_obs["fov_name"] = categorical_obs["fov_name"].astype("category")
    filtered, audit = filter_observations(categorical_obs, categorical, "plate", "GFP")
    assert len(filtered) == 3 and audit["n_unknown"] == 1
    assert eligibility_records_sha256(categorical) == eligibility_records_sha256(eligibility)
    categorical.loc[0, "reason"] = pd.NA
    _, audit = filter_observations(categorical_obs, categorical, "plate", "GFP")
    assert audit["excluded_reason_counts"] == {"<unspecified>": 1}


@pytest.mark.parametrize("all_excluded", [False, True])
@pytest.mark.parametrize("conflicting_output", [False, True])
def test_predict_run_filters_tracks_and_writer_receives_final_audit(
    obs, eligibility, tmp_path, monkeypatch, all_excluded, conflicting_output
):
    from lightning import pytorch

    from viscy_data.triplet import TripletDataModule
    from viscy_utils.callbacks import embedding_writer

    eligibility.loc[0, "reason"] = "no organelle / reviewed"
    if all_excluded:
        eligibility = obs.assign(dataset="plate", channel_name="GFP", eligible=False, reason="reviewed")
    tables = [obs.reset_index(drop=True)]
    positions = [SimpleNamespace(zgroup=SimpleNamespace(name="/A/1/000"))]
    captured = {}

    def base_init(self, **kwargs):
        captured["datamodule_kwargs"] = kwargs

    monkeypatch.setattr(TripletDataModule, "__init__", base_init)
    monkeypatch.setattr(TripletDataModule, "_align_tracks_tables_with_positions", lambda self: (positions, tables))

    def build_model(*args):
        assert not all_excluded, "Model construction occurred for a fully excluded reporter"
        assert json.loads(captured["metadata"]["phenotyping_eligibility"])["n_excluded"] == 1
        return object()

    monkeypatch.setattr(predict_triplet, "_build_dynaclr_module", build_model)

    class Writer:
        def __init__(self, **kwargs):
            assert not all_excluded, "Writer construction occurred for a fully excluded reporter"
            captured["writer_kwargs"] = kwargs

    class Trainer:
        def __init__(self, **kwargs):
            assert not all_excluded, "GPU trainer construction occurred for a fully excluded reporter"

        def predict(self, model, datamodule, return_predictions):
            _, selected = datamodule._align_tracks_tables_with_positions()
            captured["selected"] = selected[0]
            # This is the first point at which prediction may request patches.
            assert selected[0]["t"].tolist() == [1, 2, 3]
            metadata = captured["writer_kwargs"]["uns_metadata"]["phenotyping_eligibility"]
            assert json.loads(metadata)["n_excluded"] == 1

    monkeypatch.setattr(embedding_writer, "EmbeddingWriter", Writer)
    monkeypatch.setattr(pytorch, "Trainer", Trainer)
    entry = SimpleNamespace(
        experiment="plate",
        channel="GFP",
        data_path="unused",
        tracks_path="unused",
        wells=None,
        exclude_fovs=None,
        blank_qc_channels=["GFP", "Phase3D"],
        output_path=tmp_path / "out.zarr",
    )
    captured["metadata"] = {"model_family": "test"}
    kwargs = dict(
        model_type="dynaclr",
        checkpoint="unused",
        training_config=None,
        encoder_kwargs={},
        example_input_array_shape=[1, 1, 1, 16, 16],
        z_range=(0, 1),
        z_extraction_window=None,
        focus_channel=None,
        z_focus_offset=0.3,
        z_reduction="mip",
        reference_pixel_size=None,
        reference_pixel_size_z_um=None,
        yx_patch_size=(16, 16),
        katamari_model_input_size=(16, 16),
        katamari_readout="avg",
        katamari_use_teacher=False,
        batch_size=4,
        num_workers=0,
        uns_metadata=captured["metadata"],
        eligibility_table=eligibility,
        eligibility_provenance={"eligibility_file_sha256": "authoritative"},
    )
    # Eligibility uses a copied metadata dict; capture it at alignment time.
    original_alignment = EligibilityTripletMixin._align_tracks_tables_with_positions

    def capture_alignment(self):
        captured["metadata"] = self.eligibility_metadata
        return original_alignment(self)

    monkeypatch.setattr(EligibilityTripletMixin, "_align_tracks_tables_with_positions", capture_alignment)
    if conflicting_output:
        ad.AnnData(X=np.ones((1, 2), dtype=np.float32)).write_zarr(entry.output_path)
        with pytest.raises(ValueError, match="distinct run/output identity"):
            predict_triplet._run_predict(entry, **kwargs)
        assert "datamodule_kwargs" not in captured
        np.testing.assert_array_equal(ad.read_zarr(entry.output_path).X, np.ones((1, 2)))
        return
    if all_excluded:
        with pytest.raises(ValueError, match="excluded all.*no image patches"):
            predict_triplet._run_predict(entry, **kwargs)
        assert not entry.output_path.exists()
        return
    predict_triplet._run_predict(entry, **kwargs)
    audit = json.loads(captured["writer_kwargs"]["uns_metadata"]["phenotyping_eligibility"])
    assert audit["eligibility_file_sha256"] == "authoritative"
    assert audit["eligibility_records_sha256"] == eligibility_records_sha256(eligibility)
    assert captured["writer_kwargs"]["overwrite"] is True
    assert captured["datamodule_kwargs"]["blank_qc_channels"] == ["GFP", "Phase3D"]
    pd.testing.assert_frame_equal(tables[0], obs.reset_index(drop=True))
    embedding_writer.write_embedding_dataset(
        output_path=entry.output_path,
        features=np.zeros((3, 2), dtype=np.float32),
        index_df=captured["selected"],
        uns_metadata=captured["writer_kwargs"]["uns_metadata"],
    )
    persisted = json.loads(ad.read_zarr(entry.output_path).uns["phenotyping_eligibility"])
    assert persisted["excluded_reason_counts"] == {"no organelle / reviewed": 1}
