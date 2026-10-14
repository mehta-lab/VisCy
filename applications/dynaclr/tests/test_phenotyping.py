"""Behavior tests for cell attribution and scientifically safe checkpointing."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from dynaclr.phenotyping.algorithms import (
    FrameSkip,
    cytoplasm_labels,
    map_nuclei_to_cells,
    project,
    projection_window,
    resolve_track_labels,
    summarize_tracks,
    validate_tracks,
    verify_phase_alignment,
)
from dynaclr.phenotyping.annotations import compare_annotations, read_annotations
from dynaclr.phenotyping.config import Annotation, Channel, Config, Selection, channels_for_fov, load_config
from dynaclr.phenotyping.io import ensure_labels, read_position, validate_alignment
from dynaclr.phenotyping.measurements import cell_centers, measure_cells, observation_measurements
from dynaclr.phenotyping.pipeline import main, run_pipeline, select_frames


def track_rows(t=2):
    return pd.DataFrame(
        dict(
            track_id=[9, 44],
            t=[t, t],
            id=[(t + 1) * 1_000_000 + 1, (t + 1) * 1_000_000 + 2],
            y=[4.0, 8.0],
            x=[4.0, 10.0],
            parent_track_id=[-1, 9],
            parent_id=[-1, -1],
        )
    )


def masks():
    nuclei = np.zeros((16, 20), np.uint32)
    nuclei[4:6, 4:6] = 9
    nuclei[8:10, 10:12] = 44
    cells = np.zeros_like(nuclei)
    cells[2:14, 2:18] = 8001
    return cells, nuclei


def test_asymmetric_exact_five_planes_and_invalid_focus():
    window = projection_window(3, 8)
    assert (window.start, window.stop) == (2, 7)
    stack = np.broadcast_to(np.arange(8)[:, None, None], (8, 4, 6))
    np.testing.assert_array_equal(project(stack[window]), 6)
    for value in (None, True, "bad", 0, 7, 1.5, np.nan):
        with pytest.raises(FrameSkip):
            projection_window(value, 8)
    with pytest.raises(FrameSkip):
        project(np.zeros((4, 4, 4)))


def test_phase_alignment_fails_closed_and_uninformative():
    phase = np.arange(120).reshape(5, 4, 6)
    verify_phase_alignment(phase, phase.copy())
    with pytest.raises(FrameSkip, match="alignment_failed"):
        verify_phase_alignment(phase, phase + 1)
    with pytest.raises(FrameSkip, match="uninformative"):
        verify_phase_alignment(np.zeros((5, 4, 6)), np.zeros((5, 4, 6)))


def test_sparse_track_labels_verified_from_centroids_not_encoded_id():
    cells, nuclei = masks()
    verified = resolve_track_labels(nuclei, track_rows())
    assert verified.nucleus_label.tolist() == [9, 44]
    assert verified.label_encoding.tolist() == ["track_id", "track_id"]
    # Sets alone match after swapping centroids, but the footprint identity fails.
    swapped = track_rows()
    swapped[["y", "x"]] = swapped[["y", "x"]].iloc[::-1].to_numpy()
    with pytest.raises(FrameSkip, match="unverified"):
        resolve_track_labels(nuclei, swapped)
    nuclei[1, 1] = 777
    with pytest.raises(FrameSkip, match="unverified"):
        resolve_track_labels(nuclei, track_rows())


def test_detection_modulo_only_when_time_prefix_verified():
    _, nuclei = masks()
    nuclei[nuclei == 9] = 1
    nuclei[nuclei == 44] = 2
    assert resolve_track_labels(nuclei, track_rows()).label_encoding.tolist() == ["id_modulo"] * 2
    bad = track_rows()
    bad["id"] += 1_000_000
    with pytest.raises(FrameSkip):
        resolve_track_labels(nuclei, bad)


def test_empty_nuclear_mask_is_missing_not_zero_signal():
    rows = resolve_track_labels(np.zeros((16, 20), np.uint32), track_rows())
    assert rows.nucleus_label.isna().all()
    assert (rows.nucleus_status == "missing_nuclear_footprint").all()


def test_track_duplicate_and_noninteger_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_tracks(pd.concat([track_rows(), track_rows()]))
    bad = track_rows().astype({"t": float})
    bad.loc[0, "t"] = 1.5
    with pytest.raises(ValueError, match="Noninteger"):
        validate_tracks(bad)


def test_multinucleation_strict_majority_and_all_footprints_excluded():
    cells, nuclei = masks()
    mapped = map_nuclei_to_cells(cells, nuclei, resolve_track_labels(nuclei, track_rows()))
    assert mapped.cell_id.tolist() == [8001, 8001]
    assert mapped.nuclei_in_cell.tolist() == [2, 2]
    assert mapped.parent_track_id.tolist() == [-1, 9]
    cyto = cytoplasm_labels(cells, nuclei)
    assert not cyto[nuclei > 0].any()
    split = cells.copy()
    split[4:5, 4:6] = 2
    mapped = map_nuclei_to_cells(split, nuclei, resolve_track_labels(nuclei, track_rows()))
    assert mapped.loc[0, "assignment_status"] == "ambiguous"
    assert pd.isna(mapped.loc[0, "cell_id"])
    assert mapped.loc[0, "overlap_fraction"] == 0.5


def test_cp_measure_known_shape_intensity_sparse_ids_and_patch_extent():
    cells, nuclei = masks()
    cyto = cytoplasm_labels(cells, nuclei)
    membership = map_nuclei_to_cells(cells, nuclei, resolve_track_labels(nuclei, track_rows()))
    pixels = np.full(cells.shape, 7, np.float32)
    pixels[nuclei > 0] = 100
    channels = (Channel("GFP", "TOMM20"),)
    geometry, intensity = measure_cells(
        cells, cyto, nuclei, nuclei.astype(float), {"GFP": pixels}, channels, membership, (0.2, 0.3)
    )
    assert geometry.cell_id.tolist() == [8001]
    assert geometry.area_px.tolist() == [192]
    assert geometry.area_um2.iloc[0] == pytest.approx(11.52)
    assert geometry.cp_Area.iloc[0] == 192
    assert geometry.cytoplasm_area_px.iloc[0] == 184
    cyto_values = intensity[intensity.compartment == "cytoplasm"]
    assert cyto_values.mean_intensity.iloc[0] == 7
    assert cyto_values.cp_Intensity_IntegratedIntensity.iloc[0] == 184 * 7
    obs = observation_measurements(membership, geometry, intensity, channels)
    assert obs.nuclear_centered_patch_side_px.tolist() == [27, 17]
    assert obs.eligible.isna().all() and obs.below_threshold.isna().all()
    assert obs.reason.tolist() == ["no_threshold", "no_threshold"]
    assert obs.mean_intensity.tolist() == [7, 7]


def test_fully_nuclear_cell_has_no_valid_cytoplasmic_measurement():
    _, nuclei = masks()
    cells = np.where(nuclei > 0, 8001, 0).astype(np.uint32)
    membership = map_nuclei_to_cells(cells, nuclei, resolve_track_labels(nuclei, track_rows()))
    channels = (Channel("GFP", "TOMM20"),)
    geometry, intensity = measure_cells(
        cells,
        cytoplasm_labels(cells, nuclei),
        nuclei,
        nuclei.astype(float),
        {"GFP": np.ones(cells.shape)},
        channels,
        membership,
        (0.2, 0.2),
    )
    obs = observation_measurements(membership, geometry, intensity, channels)
    assert not obs.valid_signal.any() and obs.eligible.isna().all()
    assert (obs.reason == "missing").all()


def test_track_denominator_missing_frames_break_low_intervals_and_unknown():
    expected = pd.DataFrame(dict(dataset=["d"] * 5, fov=["A/1/0000"] * 5, track_id=[9] * 5, t=[0, 1, 2, 3, 4]))
    obs = expected.iloc[[0, 1, 3, 4]].copy()
    obs["marker"] = "TOMM20"
    obs["valid_signal"] = [True, True, True, False]
    obs["below_threshold"] = pd.array([True, True, False, None], dtype="boolean")
    summary = summarize_tracks(obs, expected).iloc[0]
    assert summary.coverage == 0.6
    assert summary.below_fraction == pytest.approx(2 / 3)
    assert bool(summary.mostly_low_signal) is True
    assert summary.longest_low_interval_frames == 2
    obs["below_threshold"] = pd.array([None] * 4, dtype="boolean")
    summary = summarize_tracks(obs, expected).iloc[0]
    assert pd.isna(summary.below_fraction) and pd.isna(summary.mostly_low_signal)


def _write_position(path, data, channels, *, scale_t=1800, focus=None, translate_x=0):
    import zarr

    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    transforms = [{"type": "scale", "scale": [scale_t, 1, 0.17, 0.2, 0.3]}]
    if translate_x:
        transforms.append({"type": "translation", "translation": [0, 0, 0, 0, translate_x]})
    group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "t", "type": "time", "unit": "second"},
                    {"name": "c", "type": "channel"},
                    *[{"name": a, "type": "space", "unit": "micrometer"} for a in ("z", "y", "x")],
                ],
                "datasets": [{"path": "0", "coordinateTransformations": transforms}],
            }
        ],
        "omero": {"channels": [{"label": name} for name in channels]},
    }
    if focus is not None:
        group.attrs["focus_slice"] = {"Phase3D": {"per_timepoint": focus}}
    group.create_array("0", data=data, chunks=(1, 1, 1, *data.shape[-2:]), dimension_names=("t", "c", "z", "y", "x"))
    return group


@pytest.fixture
def dataset(tmp_path):
    import zarr

    source, focus, tracking = (tmp_path / name for name in ("source.zarr", "focus.zarr", "tracking.zarr"))
    fov = "A/1/0000"
    channels = ["Phase3D", "membrane_prediction", "nuclei_prediction", "GFP"]
    cells, nuclei = masks()
    data = np.ones((4, 4, 7, 16, 20), np.float32)
    for t in range(4):
        data[t, 0] = np.arange(7 * 16 * 20).reshape(7, 16, 20) + t
        data[t, 1] = cells > 0
        data[t, 2] = nuclei > 0
        data[t, 3] *= 7
        data[t, 3, :, nuclei > 0] = 100
    for root in (source, focus, tracking):
        group = zarr.open_group(str(root), mode="w", zarr_format=3)
        group.attrs["ome"] = {
            "version": "0.5",
            "plate": {
                "name": "test",
                "rows": [{"name": "A"}],
                "columns": [{"name": "1"}],
                "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0}],
            },
        }
        group.require_group("A/1").attrs["ome"] = {"version": "0.5", "well": {"images": [{"path": "0000"}]}}
    _write_position(source / fov, data, channels)
    _write_position(
        focus / fov, data[:, [0, 3]].copy(), ["Phase3D", "GFP"], scale_t=1, focus={"0": 0, "1": 2, "2": 2, "3": 6}
    )
    labeldata = np.zeros((4, 1, 1, 16, 20), np.uint32)
    labeldata[1:3, 0, 0] = nuclei
    _write_position(tracking / fov, labeldata, ["nuclei_prediction_labels"], scale_t=1)
    tracks = pd.concat([track_rows(t) for t in range(4)], ignore_index=True)
    tracks.to_csv(tracking / fov / "tracks.csv", index=False)
    config = Config(
        "test",
        str(source),
        str(focus),
        str(tracking),
        str(tmp_path / "out"),
        fluorescence=(Channel("GFP", "TOMM20"),),
        max_overlays=0,
    )
    return config, cells, tracks


class FakeSegmenter:
    calls = 0

    def __init__(self, config):
        self.identity = {"checkpoint": "synthetic", "backbone": "dino_vitl"}

    def __call__(self, image, pixels):
        type(self).calls += 1
        return masks()[0]


def test_coordinate_mismatch_and_hcs_native_label_metadata(dataset):
    config, _, _ = dataset
    source = read_position(Path(config.source) / "A/1/0000")
    focus = read_position(Path(config.focus_source) / "A/1/0000")
    tracking = read_position(Path(config.tracks) / "A/1/0000")
    validate_alignment(source, focus, tracking)
    with pytest.raises(FrameSkip, match="origin"):
        validate_alignment(source, replace(focus, translation=(0, 0, 0, 0, 3)), tracking)
    with pytest.raises(FrameSkip, match="shape"):
        validate_alignment(source, replace(focus, shape=(4, 2, 6, 16, 20)), tracking)
    arrays = ensure_labels(Path(config.output), "A/1/0000", source)
    assert arrays["cells"].shape == (4, 1, 1, 16, 20)
    assert arrays["cells"].dtype == np.uint32
    import zarr

    root = zarr.open_group(str(Path(config.output) / "segmentations.ome.zarr"), mode="r")
    assert root.attrs["ome"]["plate"]["wells"][0]["path"] == "A/1"
    assert root["A/1"].attrs["ome"]["well"]["images"] == [{"path": "0000"}]


def test_identity_transform_is_noop_and_unknown_transform_rejected(tmp_path):
    path = tmp_path / "tracking.zarr"
    group = _write_position(path, np.zeros((2, 1, 1, 4, 5), np.uint32), ["labels"])
    ome = group.attrs["ome"]
    ome["multiscales"][0]["coordinateTransformations"] = [{"type": "identity"}]
    group.attrs["ome"] = ome
    position = read_position(path)
    assert position.scale == (1800, 1, 0.17, 0.2, 0.3)
    assert position.translation == (0, 0, 0, 0, 0)
    dataset_transforms = ome["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    ome["multiscales"][0]["datasets"][0]["coordinateTransformations"] = []
    group.attrs["ome"] = ome
    with pytest.raises(FrameSkip, match="missing_coordinate_scale"):
        read_position(path)
    ome["multiscales"][0]["datasets"][0]["coordinateTransformations"] = dataset_transforms
    ome["multiscales"][0]["coordinateTransformations"] = [{"type": "rotation", "rotation": [[1, 0], [0, 1]]}]
    group.attrs["ome"] = ome
    with pytest.raises(FrameSkip, match="unsupported_coordinate_transform"):
        read_position(path)


def test_staged_pipeline_completion_resume_skip_status_and_provenance(dataset):
    config, cells, _ = dataset
    FakeSegmenter.calls = 0
    selected = replace(config, selection=Selection(times=(1,)))
    first = run_pipeline(selected, segmenter_factory=FakeSegmenter)
    assert not first["full_dataset_complete"] and FakeSegmenter.calls == 1
    full = run_pipeline(config, resume=True, segmenter_factory=FakeSegmenter)
    assert full["full_dataset_complete"] and FakeSegmenter.calls == 2
    output = Path(config.output)
    status = pd.read_parquet(output / "status.parquet")
    assert status.state.tolist() == ["skipped", "complete", "complete", "skipped"]
    assert set(status.skip_reason.dropna()) == {"focus_window_out_of_bounds"}
    eligibility = pd.read_parquet(output / "eligibility.parquet")
    assert len(eligibility) == 8 and eligibility.eligible.isna().all()
    assert eligibility.reason.value_counts().to_dict() == {"missing": 4, "no_threshold": 4}
    geometries = pd.read_parquet(output / "cells.parquet")
    assert len(geometries) == 2  # not multiplied by two nuclear tracks
    assert geometries.multinucleated.all()
    coverage = pd.read_parquet(output / "track_summaries.parquet")
    assert coverage.coverage.tolist() == [0.5, 0.5]
    assert coverage.mostly_low_signal.isna().all()
    import zarr

    labels = zarr.open_group(str(output / "segmentations.ome.zarr"), mode="r")["A/1/0000/0"]
    np.testing.assert_array_equal(labels[1, 0, 0], cells)
    assert (output / "provenance/source/pipeline.py").exists()
    assert run_pipeline(config, resume=True, segmenter_factory=FakeSegmenter)["full_dataset_complete"]
    assert FakeSegmenter.calls == 2
    with pytest.raises(RuntimeError, match="provenance"):
        run_pipeline(
            replace(config, segmentation=replace(config.segmentation, target_pixel_size_um=0.58)),
            resume=True,
            segmenter_factory=FakeSegmenter,
        )


def test_dry_run_no_output_or_model_init(dataset, tmp_path):
    config, _, _ = dataset
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            dict(
                dataset=config.dataset,
                source=config.source,
                focus_source=config.focus_source,
                tracks=config.tracks,
                output=config.output,
                channels={"fluorescence": [{"name": "GFP", "marker": "TOMM20"}]},
            )
        )
    )
    result = CliRunner().invoke(main, ["--config", str(path), "--time", "1,2", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not Path(config.output).exists()


def test_mixed_well_marker_resolution_and_overlapping_selectors_rejected(tmp_path):
    raw = dict(
        dataset="mixed",
        source="s",
        focus_source="f",
        tracks="t",
        output="o",
        channels={
            "fluorescence": [
                {"name": "GFP", "marker": "TOMM20", "wells": ["A/3", "B/3"]},
                {"name": "GFP", "marker": "SEC61B", "wells": ["A/2", "B/2"]},
            ]
        },
    )
    path = tmp_path / "c.yml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    assert channels_for_fov(config, "B/3/0000")[0].marker == "TOMM20"
    assert channels_for_fov(config, "A/2/0000")[0].marker == "SEC61B"
    raw["channels"]["fluorescence"][1]["wells"] = ["A/3"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="disjoint"):
        load_config(path)


def test_annotations_obs_only_snapshot_and_identity_mismatch(tmp_path):
    import zarr
    from anndata.io import write_elem

    root = zarr.open_group(str(tmp_path / "ann.zarr"), mode="w", zarr_format=2)
    obs = pd.DataFrame(
        dict(
            dataset=["short"] * 2,
            fov_name=["A/1/0000"] * 2,
            track_id=[9, 44],
            t=[2, 2],
            id=[3000001, 3000002],
            empty=["empty", None],
        ),
        index=["a", "b"],
    )
    write_elem(root, "obs", obs)
    # If the reader touches X it would fail; no expression/embedding arrays exist.
    annotations, identity = read_annotations((Annotation(str(tmp_path / "ann.zarr"), "test", "confident"),))
    assert identity[0]["empty_rows"] == 1 and annotations.annotated_empty.isna().sum() == 1
    measured = pd.DataFrame(
        dict(
            dataset=["test"] * 2,
            fov=["A/1/0000"] * 2,
            track_id=[9, 44],
            t=[2, 2],
            id=[3000099, 3000002],
            marker=["TOMM20"] * 2,
            channel_name=["GFP"] * 2,
            assignment_status=["matched"] * 2,
            mean_intensity=[7.0, 100.0],
            median_intensity=[7.0, 100.0],
            p90_intensity=[7.0, 100.0],
            valid_signal=[True] * 2,
        )
    )
    comparison = compare_annotations(annotations, measured)
    assert comparison.annotation_match_status.tolist() == ["identity_mismatch", "matched"]
    assert np.isnan(comparison.mean_intensity.iloc[0])


def test_annotation_enrichment_uses_effective_dataset():
    config = Config("april", "s", "f", "t", "o", selection=Selection(annotation_enriched=True))
    inventory = {"selected": [{"fov": "A/1/0000", "shape": [4, 1, 7, 16, 20], "timepoints": [0, 1, 2, 3]}]}
    annotations = pd.DataFrame(
        dict(dataset=["april", "july"], fov=["A/1/0000"] * 2, t=[1, 3], annotated_empty=[True, True])
    )
    assert select_frames(config, inventory, annotations)["selected"][0]["timepoints"] == [1]


def test_strict_cpdino_backbone_rejects_sam_and_vitb(monkeypatch):
    from cellpose import models, vit

    from dynaclr.phenotyping.segmentation import verify_model

    class Net:
        def state_dict(self):
            return {"encoder.cls_token": np.zeros(1), "encoder.patch_embed.proj.weight": np.zeros((1024, 1))}

    monkeypatch.setattr(vit, "CPDINO", Net)
    monkeypatch.setattr(models, "get_backbone", lambda path: "dino_vitl")
    model = SimpleNamespace(backbone="dino_vitl", net=Net())
    verify_model(model, "fake")
    for backbone in ("sam_vitl", "dino_vitb"):
        with pytest.raises(RuntimeError):
            verify_model(SimpleNamespace(backbone=backbone, net=Net()), "fake")
    monkeypatch.setattr(models, "get_backbone", lambda path: "sam_vitl")
    with pytest.raises(RuntimeError):
        verify_model(model, "fake")


def test_cpdino_raw_grayscale_native_and_explicit_downsample():
    from dynaclr.phenotyping.config import Segmentation
    from dynaclr.phenotyping.segmentation import CPDinoSegmenter

    class Model:
        def __init__(self):
            self.inputs = []

        def eval(self, image, **kwargs):
            self.inputs.append((image.copy(), kwargs))
            labels = np.zeros(image.shape, np.uint32)
            labels[1:-1, 1:-1] = 70000
            return labels, None, None

    segmenter = object.__new__(CPDinoSegmenter)
    segmenter.model = Model()
    segmenter.config = Segmentation()
    raw = np.arange(48 * 52, dtype=np.float32).reshape(48, 52) + 1000
    output = segmenter(raw, (0.2, 0.2))
    image, kwargs = segmenter.model.inputs[-1]
    np.testing.assert_array_equal(image, raw)
    assert image.ndim == 2 and kwargs["channel_axis"] is None and kwargs["normalize"] is True
    assert output.dtype == np.uint32 and output.shape == raw.shape and output.max() == 70000
    segmenter.config = Segmentation(target_pixel_size_um=0.58)
    output = segmenter(raw, (0.2, 0.2))
    image, kwargs = segmenter.model.inputs[-1]
    assert image.shape == (32, 32) and image.ndim == 2 and image.min() > 900
    assert output.shape == raw.shape and set(np.unique(output)) == {0, 70000}


def test_cell_centers_sparse_hollow_asymmetric_and_empty_integration():
    cells = np.zeros((16, 24), np.uint32)
    cells[1:6, 2:9] = 7
    cells[2:5, 3:8] = 0  # hollow cell; bbox center lies in its hole
    cells[7:12, 2:4] = 8001
    cells[11, 4:10] = 8001  # asymmetric L
    cells[1:5, 14:20] = 1_000_003
    nuclei = np.zeros_like(cells)
    nuclei[1, 2] = 100
    nuclei[7:9, 2] = 44
    nuclei[11, 8:10] = 66
    nuclei[1:5, 14:20] = 777  # this entire cell has no cytoplasm
    nuclei[7, 18] = 900  # tracked nucleus without a membrane cell
    tracks = pd.DataFrame(
        dict(
            track_id=[100, 44, 66, 777, 900],
            t=[2] * 5,
            id=[3_000_001, 3_000_002, 3_000_003, 3_000_004, 3_000_005],
            y=[1.0, 7.5, 11.0, 2.5, 7.0],
            x=[2.0, 2.0, 8.5, 16.5, 18.0],
            parent_track_id=[-1] * 5,
            parent_id=[-1] * 5,
        )
    )
    membership = map_nuclei_to_cells(cells, nuclei, resolve_track_labels(nuclei, tracks))
    cytoplasm = cytoplasm_labels(cells, nuclei)
    cols = ["bbox_center_y", "bbox_center_x", "cytoplasm_centroid_y", "cytoplasm_centroid_x"]
    expected = np.array([[3.0, 5.0, 59 / 19, 98 / 19], [9.0, 5.5, 119 / 12, 43 / 12], [2.5, 16.5, np.nan, np.nan]])
    centers = cell_centers(cells, cytoplasm)
    assert centers.cell_id.tolist() == [7, 8001, 1_000_003]
    np.testing.assert_allclose(centers[cols], expected, equal_nan=True)
    assert cells[3, 5] == 0
    channels = (Channel("GFP", "TOMM20"),)
    pixels = np.ones(cells.shape, np.float32)
    geometry, intensity = measure_cells(
        cells,
        cytoplasm,
        nuclei,
        (nuclei > 0).astype(float),
        {"GFP": pixels},
        channels,
        membership,
        (0.2, 0.3),
    )
    np.testing.assert_allclose(geometry[cols], expected, equal_nan=True)
    obs = observation_measurements(membership, geometry, intensity, channels)
    np.testing.assert_array_equal(obs[["y", "x"]], tracks[["y", "x"]])
    np.testing.assert_array_equal(obs[["nucleus_center_y", "nucleus_center_x"]], tracks[["y", "x"]])
    shared = obs[obs.track_id.isin([44, 66])]
    np.testing.assert_allclose(shared[cols], np.tile(expected[1], (2, 1)))
    assert shared.nuclei_in_cell.tolist() == [2, 2]
    assert shared.nucleus_center_y.tolist() == [7.5, 11.0]
    unmatched = obs.loc[obs.track_id.eq(900)].iloc[0]
    assert unmatched.assignment_status == "unmatched" and unmatched[cols].isna().all()
    fully_nuclear = obs.loc[obs.track_id.eq(777)].iloc[0]
    assert fully_nuclear[["cytoplasm_centroid_y", "cytoplasm_centroid_x"]].isna().all()
    assert fully_nuclear.bbox_center_x == 16.5
    changed_pixels = np.arange(cells.size, dtype=float).reshape(cells.shape) ** 2
    changed_geometry, changed_intensity = measure_cells(
        cells,
        cytoplasm,
        nuclei,
        (nuclei > 0).astype(float),
        {"GFP": changed_pixels},
        channels,
        membership,
        (0.2, 0.3),
    )
    np.testing.assert_allclose(changed_geometry[cols], expected, equal_nan=True)
    assert not np.array_equal(intensity.mean_intensity, changed_intensity.mean_intensity)


def test_cell_centers_all_background_keep_observation_schema():
    cells = np.zeros((16, 20), np.uint32)
    _, nuclei = masks()
    membership = map_nuclei_to_cells(cells, nuclei, resolve_track_labels(nuclei, track_rows()))
    channels = (Channel("GFP", "TOMM20"),)
    geometry, intensity = measure_cells(
        cells,
        cells.copy(),
        nuclei,
        nuclei.astype(float),
        {"GFP": np.ones(cells.shape)},
        channels,
        membership,
        (0.2, 0.3),
    )
    cols = ["bbox_center_y", "bbox_center_x", "cytoplasm_centroid_y", "cytoplasm_centroid_x"]
    centers = cell_centers(cells, cells.copy())
    assert centers.empty and centers.columns.tolist() == ["cell_id", *cols]
    pd.testing.assert_frame_equal(geometry, centers)
    obs = observation_measurements(membership, geometry, intensity, channels)
    assert obs[cols].isna().all().all()
    np.testing.assert_array_equal(obs[["nucleus_center_y", "nucleus_center_x"]], track_rows()[["y", "x"]])


def test_cell_centers_reject_coordinate_or_label_mismatch():
    cells, nuclei = masks()
    with pytest.raises(ValueError, match="native 2D grid"):
        cell_centers(cells, cells[:-1])
    bad_cyto = cytoplasm_labels(cells, nuclei)
    bad_cyto[3, 3] = 123
    with pytest.raises(ValueError, match="owning cell label"):
        cell_centers(cells, bad_cyto)
