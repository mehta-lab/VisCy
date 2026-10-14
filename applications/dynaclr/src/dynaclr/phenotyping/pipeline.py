"""Staged CP-DINO segmentation. Distinct frame/FOV shards share one output.

Summarize after shards finish; dry-run reads metadata without writes/model load.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import click
import numpy as np
import pandas as pd

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
from dynaclr.phenotyping.annotations import candidate_agreement, compare_annotations, read_annotations
from dynaclr.phenotyping.config import Config, Selection, channels_for_fov, load_config, normalize_fov
from dynaclr.phenotyping.io import (
    assert_compute_node,
    atomic_json,
    atomic_parquet,
    code_identity,
    ensure_labels,
    find_track_csv,
    lock,
    positions,
    read_position,
    snapshot_code,
    source_identity,
    stable_hash,
    validate_alignment,
)
from dynaclr.phenotyping.measurements import measure_cells, observation_measurements

FORMAT_VERSION = 1


def _now():
    return datetime.now(timezone.utc).isoformat()


def inspect_inputs(config: Config) -> dict:
    """Inspect dataset metadata and selected work without loading image data."""
    source, focus, tracks = [positions(getattr(config, key)) for key in ("source", "focus_source", "tracks")]
    excluded = set(config.exclude_fovs)
    selected = list(config.selection.fovs) or sorted(set(source) - excluded)
    if set(selected) - set(source):
        raise ValueError(f"FOVs absent from assembled source: {sorted(set(selected) - set(source))}")
    records = []
    for fov in sorted(set(selected) - excluded):
        try:
            position = read_position(Path(config.source) / fov)
            times = list(config.selection.times) or list(range(position.shape[0]))
            if any(t < 0 or t >= position.shape[0] for t in times):
                raise ValueError(f"Time selection outside {fov} T={position.shape[0]}")
            records.append(
                {
                    "fov": fov,
                    "timepoints": sorted(set(times)),
                    "shape": list(position.shape),
                    "focus_available": fov in focus,
                    "tracks_available": fov in tracks,
                    "fluorescence": [asdict(ch) for ch in channels_for_fov(config, fov)],
                }
            )
        except FrameSkip as exc:
            records.append({"fov": fov, "skip_reason": str(exc), "timepoints": []})
    return {
        "dataset": config.dataset,
        "output": config.output,
        "source_positions": len(source),
        "focus_positions": len(focus),
        "tracking_positions": len(tracks),
        "source_without_tracks": sorted(set(source) - set(tracks)),
        "tracking_without_source": sorted(set(tracks) - set(source)),
        "source_without_focus": sorted(set(source) - set(focus)),
        "excluded_fovs": sorted(excluded),
        "selected": records,
        "annotation_enriched": config.selection.annotation_enriched,
    }


def select_frames(config: Config, inventory: dict, annotations: pd.DataFrame) -> dict:
    """Resolve annotated frame enrichment for the effective dataset only."""
    selected = []
    for position in inventory["selected"]:
        record = dict(position)
        marked = (
            annotations[
                (annotations.dataset == config.dataset)
                & (annotations.fov == record["fov"])
                & annotations.annotated_empty.fillna(False)
            ]
            if len(annotations)
            else pd.DataFrame()
        )
        if config.selection.annotation_enriched:
            times = set(config.selection.times) | (set(marked.t.astype(int)) if len(marked) else set())
            if "shape" in record and any(t < 0 or t >= record["shape"][0] for t in times):
                raise ValueError(f"Annotation times outside {record['fov']}")
            record["timepoints"] = sorted(times)
            if not times and "skip_reason" not in record:
                continue
        selected.append(record)
    return dict(inventory, selected=selected)


def initialize(config, config_path, annotations, annotation_ids, inventory, resume):
    """Create or validate the attempt manifest and immutable source snapshot."""
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    with lock(output / "locks" / "manifest.lock"):
        code = code_identity()
        scientific = {
            "format_version": FORMAT_VERSION,
            "config": config.scientific_dict(),
            "source": source_identity(config),
            "annotations": annotation_ids,
            "source_sha256": code["source_sha256"],
            "versions": code["versions"],
            "python": code["python"],
        }
        fingerprint = stable_hash(scientific)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["fingerprint"] != fingerprint:
                raise RuntimeError(
                    "Resume provenance differs (code/settings/inputs/environment); use a new numbered attempt"
                )
            if not resume:
                raise RuntimeError("Output is initialized; pass --resume")
        else:
            if (output / "segmentations.ome.zarr").exists() or (output / "frames").exists():
                raise RuntimeError("Image/frame outputs have no manifest; refusing unverified results")
            snapshot_code(output, config_path)
            atomic_parquet(output / "provenance" / "annotation_obs.parquet", annotations)
            manifest = {
                "fingerprint": fingerprint,
                "scientific": scientific,
                "code": code,
                "started_at": _now(),
                "state": "initialized",
                "selections": [],
            }
        manifest["selections"].append({"at": _now(), "pid": os.getpid(), "selection": asdict(config.selection)})
        atomic_json(manifest_path, manifest)
        atomic_json(output / "inventory.json", inventory)
        return manifest


def _frame_path(output, fov, t):
    return output / "frames" / fov / f"t{t:06d}"


def _with_identity(frame, dataset, fov, t=None):
    frame = frame.copy()
    frame["dataset"] = dataset
    frame["fov"] = fov
    frame["fov_name"] = fov
    if t is not None:
        frame["t"] = t
    return frame


def _skipped_observations(config, fov, rows, reason):
    membership = rows.copy()
    membership["cell_id"] = pd.array([pd.NA] * len(rows), dtype="Int64")
    membership["assignment_status"] = "missing"
    membership["frame_skip_reason"] = reason
    membership["nuclei_in_cell"] = 0
    obs = observation_measurements(membership, pd.DataFrame(), pd.DataFrame(), channels_for_fov(config, fov))
    return membership, obs


def run_pipeline(
    config: Config,
    *,
    config_path: Path | None = None,
    resume=False,
    init_only=False,
    no_summary=False,
    segmenter_factory: Callable | None = None,
):
    """Process selected frames with verified coordinates and atomic completion markers."""
    assert_compute_node()
    annotations, annotation_ids = read_annotations(config.annotations)
    inventory = select_frames(config, inspect_inputs(config), annotations)
    manifest = initialize(config, config_path, annotations, annotation_ids, inventory, resume)
    output = Path(config.output)
    segmenter = None
    overlay_count = 0
    for entry in inventory["selected"]:
        fov = entry["fov"]
        source = focus = tracking = None
        tracks = pd.DataFrame(columns=["t", "track_id", "id", "y", "x"])
        failure = entry.get("skip_reason")
        try:
            source = read_position(Path(config.source) / fov)
            focus = read_position(Path(config.focus_source) / fov)
            tracking = read_position(Path(config.tracks) / fov)
            validate_alignment(source, focus, tracking)
            required = {config.membrane_channel, config.nuclei_channel, config.focus_channel}
            required.update(ch.name for ch in channels_for_fov(config, fov))
            if required - set(source.channels) or config.focus_channel not in focus.channels:
                raise FrameSkip("required_channel_missing")
            tracks = validate_tracks(pd.read_csv(find_track_csv(config.tracks, fov)))
            if (tracks.t >= source.shape[0]).any():
                raise FrameSkip("track_times_out_of_bounds")
        except (FileNotFoundError, FrameSkip, ValueError) as exc:
            failure = str(exc)
        if init_only:
            if failure is None:
                ensure_labels(output, fov, source)
            continue
        if failure:
            if tracks.empty:
                try:
                    tracks = validate_tracks(pd.read_csv(find_track_csv(config.tracks, fov)))
                except (FileNotFoundError, FrameSkip, ValueError):
                    pass
        else:
            arrays = ensure_labels(output, fov, source)
            source_array, focus_array, track_array = source.open(), focus.open(), tracking.open()
        for t in entry["timepoints"]:
            path = _frame_path(output, fov, t)
            with lock(output / "locks" / f"{fov.replace('/', '_')}_t{t:06d}.lock", blocking=False):
                status_path = path / "status.json"
                if status_path.exists():
                    prior = json.loads(status_path.read_text())
                    if prior.get("fingerprint") != manifest["fingerprint"]:
                        raise RuntimeError(f"Frame provenance mismatch: {path}")
                    if prior.get("state") in ("complete", "skipped"):
                        continue
                start = time.perf_counter()
                rows = tracks[tracks.t == t].copy()
                status = {
                    "dataset": config.dataset,
                    "fov": fov,
                    "fov_name": fov,
                    "t": t,
                    "fingerprint": manifest["fingerprint"],
                    "started_at": _now(),
                    "state": "running",
                }
                atomic_json(status_path, status)
                try:
                    if failure:
                        raise FrameSkip(failure)
                    focus_value = (
                        focus.attrs.get("focus_slice", {})
                        .get(config.focus_channel, {})
                        .get("per_timepoint", {})
                        .get(str(t))
                    )
                    window = projection_window(focus_value, source.shape[2])
                    phase = np.asarray(source_array[t, source.channels.index(config.focus_channel), window, :, :])
                    prepared_phase = np.asarray(
                        focus_array[t, focus.channels.index(config.focus_channel), window, :, :]
                    )
                    verify_phase_alignment(
                        phase, prepared_phase, config.phase_alignment_rtol, config.phase_alignment_atol
                    )
                    membrane = project(
                        np.asarray(source_array[t, source.channels.index(config.membrane_channel), window, :, :])
                    )
                    virtual_nucleus = project(
                        np.asarray(source_array[t, source.channels.index(config.nuclei_channel), window, :, :])
                    )
                    nuclei = np.asarray(track_array[t, 0, 0])
                    if not np.issubdtype(nuclei.dtype, np.integer) or np.any(nuclei < 0):
                        raise FrameSkip("invalid_nuclear_labels")
                    verified = resolve_track_labels(nuclei, rows, config.track_label_mode, config.track_id_stride)
                    if segmenter is None:
                        if segmenter_factory is None:
                            from dynaclr.phenotyping.segmentation import CPDinoSegmenter

                            segmenter_factory = CPDinoSegmenter
                        segmenter = segmenter_factory(config.segmentation)
                        with lock(output / "locks" / "manifest.lock"):
                            model_path = output / "provenance" / "model.json"
                            if model_path.exists() and json.loads(model_path.read_text()) != segmenter.identity:
                                raise RuntimeError("Checkpoint identity differs from this attempt")
                            atomic_json(model_path, segmenter.identity)
                    before = time.perf_counter()
                    cells = segmenter(membrane, source.pixel_size_yx_um)
                    status["segmentation_seconds"] = time.perf_counter() - before
                    if cells.shape != nuclei.shape or cells.dtype != np.uint32:
                        raise RuntimeError("Segmenter must return native XY uint32")
                    membership = map_nuclei_to_cells(cells, nuclei, verified)
                    cytoplasm = cytoplasm_labels(cells, nuclei)
                    channels = channels_for_fov(config, fov)
                    fluorescence = {
                        ch.name: project(np.asarray(source_array[t, source.channels.index(ch.name), window, :, :]))
                        for ch in channels
                    }
                    before = time.perf_counter()
                    geometry, intensity = measure_cells(
                        cells,
                        cytoplasm,
                        nuclei,
                        virtual_nucleus,
                        fluorescence,
                        channels,
                        membership,
                        source.pixel_size_yx_um,
                    )
                    obs = observation_measurements(membership, geometry, intensity, channels)
                    status["measurement_seconds"] = time.perf_counter() - before
                    for name, labels in (("cells", cells), ("cytoplasm", cytoplasm), ("nuclei", nuclei)):
                        arrays[name][t, 0, 0] = labels.astype(np.uint32)
                    for name, frame in (
                        ("cells", geometry),
                        ("intensity", intensity),
                        ("membership", membership),
                        ("observations", obs),
                    ):
                        atomic_parquet(path / f"{name}.parquet", _with_identity(frame, config.dataset, fov, t))
                    digest = hashlib.sha256()
                    for image in (phase, membrane, virtual_nucleus, nuclei, *fluorescence.values()):
                        digest.update(np.ascontiguousarray(image).tobytes())
                    status.update(
                        state="complete",
                        focus=int(focus_value),
                        z_start=window.start,
                        z_stop=window.stop,
                        instances=len(geometry),
                        track_rows=len(rows),
                        input_pixels_sha256=digest.hexdigest(),
                    )
                    marked = (
                        annotations[
                            (annotations.dataset == config.dataset) & (annotations.fov == fov) & (annotations.t == t)
                        ]
                        if len(annotations)
                        else annotations
                    )
                    if overlay_count < config.max_overlays:
                        from dynaclr.phenotyping.report import frame_overlay

                        frame_overlay(
                            path / "overlay.png",
                            membrane,
                            virtual_nucleus,
                            fluorescence,
                            cells,
                            nuclei,
                            membership,
                            marked,
                        )
                        overlay_count += 1
                except FrameSkip as exc:
                    membership, obs = _skipped_observations(config, fov, rows, str(exc))
                    atomic_parquet(path / "membership.parquet", _with_identity(membership, config.dataset, fov, t))
                    atomic_parquet(path / "observations.parquet", _with_identity(obs, config.dataset, fov, t))
                    status.update(state="skipped", skip_reason=str(exc), track_rows=len(rows))
                except Exception as exc:
                    status.update(state="failed", error=f"{type(exc).__name__}: {exc}", ended_at=_now())
                    atomic_json(status_path, status)
                    raise
                status.update(ended_at=_now(), total_seconds=time.perf_counter() - start)
                atomic_json(status_path, status)
                click.echo(f"{fov} t={t}: {status['state']} ({status['total_seconds']:.1f}s)", err=True)
    if init_only:
        return {"state": "initialized", "output": str(output), "fingerprint": manifest["fingerprint"]}
    return {"state": "selection_complete", "output": str(output)} if no_summary else summarize(config)


def summarize(config: Config) -> dict:
    """Rebuild aggregates from completed frame artifacts, never partial writes."""
    assert_compute_node()
    from dynaclr.phenotyping.report import summary_report

    output = Path(config.output)
    manifest = json.loads((output / "manifest.json").read_text())
    if code_identity()["source_sha256"] != manifest["scientific"]["source_sha256"]:
        raise RuntimeError("Summary code differs from archived source; use the snapshot or a new attempt")
    with lock(output / "locks" / "summary.lock"):
        groups = {name: [] for name in ("cells", "intensity", "membership", "observations")}
        status_rows = []
        for file in sorted((output / "frames").glob("*/*/*/t*/status.json")):
            record = json.loads(file.read_text())
            if record["fingerprint"] != manifest["fingerprint"]:
                raise RuntimeError(f"Invalid frame provenance: {file}")
            status_rows.append(record)
            if record["state"] not in ("complete", "skipped"):
                continue
            for name in groups:
                artifact = file.parent / f"{name}.parquet"
                if artifact.exists() and (record["state"] == "complete" or name in ("membership", "observations")):
                    frame = pd.read_parquet(artifact)
                    if len(frame):
                        groups[name].append(frame)
        frames = {
            name: pd.concat(values, ignore_index=True) if values else pd.DataFrame() for name, values in groups.items()
        }
        status = pd.DataFrame(status_rows)
        observations = frames["observations"]
        expected_rows = []
        for fov in positions(config.source):
            if fov in config.exclude_fovs:
                continue
            try:
                tracks = validate_tracks(pd.read_csv(find_track_csv(config.tracks, fov)))
                expected_rows.append(_with_identity(tracks, config.dataset, fov))
            except (FileNotFoundError, FrameSkip, ValueError):
                continue
        expected = (
            pd.concat(expected_rows, ignore_index=True)
            if expected_rows
            else pd.DataFrame(columns=["dataset", "fov", "track_id", "t"])
        )
        tracks = summarize_tracks(observations, expected)
        annotations = pd.read_parquet(output / "provenance" / "annotation_obs.parquet")
        comparison = compare_annotations(annotations, observations)
        cols = ["dataset", "fov_name", "track_id", "t", "channel_name", "marker", "eligible", "reason"]
        eligibility = observations[cols].copy() if len(observations) else pd.DataFrame(columns=cols)
        all_expected = []
        for fov, group in expected.groupby("fov"):
            for channel in channels_for_fov(config, fov):
                item = group[["dataset", "fov_name", "track_id", "t"]].copy()
                item["channel_name"], item["marker"] = channel.name, channel.marker
                all_expected.append(item)
        if all_expected:
            full = pd.concat(all_expected, ignore_index=True)
            eligibility = full.merge(eligibility, on=cols[:6], how="left", validate="one_to_one")
            eligibility["reason"] = eligibility.reason.fillna("missing")
            eligibility["eligible"] = pd.array(eligibility.eligible, dtype="boolean")
        for name, frame in frames.items():
            atomic_parquet(output / f"{name}.parquet", frame)
        for name, frame in (
            ("status", status),
            ("track_summaries", tracks),
            ("eligibility", eligibility),
            ("annotation_comparison", comparison),
            ("candidate_thresholds", candidate_agreement(comparison, config.candidate_thresholds)),
        ):
            atomic_parquet(output / f"{name}.parquet", frame)
        report = summary_report(output, frames["cells"], observations, comparison, status, frames["membership"])
        expected_frames = {
            (fov, t)
            for fov in positions(config.source)
            if fov not in config.exclude_fovs
            for t in range(read_position(Path(config.source) / fov).shape[0])
        }
        finished = {(row["fov"], row["t"]) for row in status_rows if row["state"] in ("complete", "skipped")}
        report["full_dataset_complete"] = expected_frames.issubset(finished)
        report["expected_dataset_frames"] = len(expected_frames)
        atomic_json(output / "summary.json", report)
        return report


def _values(options, integer=False):
    values = []
    for option in options:
        for value in option.split(","):
            if integer and ":" in value:
                parts = [int(x) for x in value.split(":")]
                if len(parts) not in (2, 3):
                    raise click.BadParameter("Time range: start:stop[:step]")
                values.extend(range(*parts))
            else:
                values.append(int(value) if integer else normalize_fov(value))
    return tuple(sorted(set(values)))


@click.command(context_settings={"show_default": True})
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dataset", default=None, help="Verify this single-dataset config's identity.")
@click.option("--output", default=None, type=click.Path(path_type=Path))
@click.option("--fov", multiple=True, help="Exact row/column/FOV; repeat or comma separate.")
@click.option("--time", "times", multiple=True, help="Zero-based frames; repeat, comma separate or start:stop[:step].")
@click.option(
    "--annotation-frames", is_flag=True, help="Alone selects annotated frames; with times includes annotated frames."
)
@click.option("--resume", is_flag=True, help="Continue an attempt only with matching scientific provenance.")
@click.option("--init", "init_only", is_flag=True, help="Initialize shared labels/metadata without inference.")
@click.option("--summarize", "summary_only", is_flag=True, help="Combine completed shards; no inference.")
@click.option("--no-summary", is_flag=True, help="Concurrent shards: summarize once after all finish.")
@click.option("--dry-run", is_flag=True, help="Read metadata without writes or model initialization.")
def main(
    config_path, dataset, output, fov, times, annotation_frames, resume, init_only, summary_only, no_summary, dry_run
):
    """Segment projected virtual membrane and measure fluorescence inside cells."""
    try:
        config = load_config(config_path, dataset, output)
        selection = config.selection
        config = replace(
            config,
            selection=Selection(
                _values(fov) if fov else selection.fovs,
                _values(times, True) if times else selection.times,
                annotation_frames or selection.annotation_enriched,
            ),
        )
        if dry_run:
            inventory = inspect_inputs(config)
            if config.selection.annotation_enriched:
                annotations, _ = read_annotations(config.annotations)
                inventory = select_frames(config, inventory, annotations)
            click.echo(json.dumps(inventory, indent=2))
        elif summary_only:
            click.echo(json.dumps(summarize(config), indent=2))
        else:
            click.echo(
                json.dumps(
                    run_pipeline(
                        config, config_path=config_path, resume=resume, init_only=init_only, no_summary=no_summary
                    ),
                    indent=2,
                )
            )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
