"""Compare CP-DINO whole-cell recipes on the 20 April frames of the cell-centers review.

Arms differ only in working grid and membrane slab. ``existing_058_5plane`` reads the
source attempt's 0.58 um labels without inference. ``native_5plane`` and ``native_slab1``
run ``CPDinoSegmenter`` at native 0.1133 um on the membrane_prediction maximum projection
over planes [focus-1, focus+4) and [focus-1, focus+2), the latter being the DynaCell eval
in-focus slab (halfwidth 1). Segmentation and tracked-nucleus joins only: no fluorescence,
no size filter, no ground-truth membrane. Reserve with ``--reserve-only`` first, then pass
the printed folder as ``--target`` so ``2>&1 | tee <target>/run.log`` logs inside it.
"""

import argparse
import inspect
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import closing
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from matplotlib.figure import Figure
from review_cytoplasm_coverage import bounds, file_sha, padded_crop
from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries

from dynaclr.phenotyping.algorithms import (
    map_nuclei_to_cells,
    project,
    projection_window,
    resolve_track_labels,
    validate_tracks,
    verify_phase_alignment,
)
from dynaclr.phenotyping.config import load_config
from dynaclr.phenotyping.io import (
    assert_compute_node,
    atomic_json,
    atomic_parquet,
    code_identity,
    find_track_csv,
    read_position,
    validate_alignment,
)
from dynaclr.phenotyping.segmentation import CPDinoSegmenter

DATASET = "2026_04_10_A549_TOMM20_ZIKV"
ARMS = ("existing_058_5plane", "native_5plane", "native_slab1")
NATIVE_ARMS = ("native_5plane", "native_slab1")
ARM_DEFINITIONS = {
    "existing_058_5plane": "source-attempt cells: MIP z[focus-1, focus+4), CP-DINO at 0.58 um, nearest-restored",
    "native_5plane": "MIP z[focus-1, focus+4) (projection_window), CP-DINO at native 0.1133 um",
    "native_slab1": "MIP z[focus-1, focus+2) (DynaCell eval slab, halfwidth 1), CP-DINO at native 0.1133 um",
}
FLOOR_UM2 = 46.5
MARKER = "TOMM20"
KEYS = ["fov", "t", "track_id"]
ZOOM = {"fov": "A/2/0001", "t": 38, "track_id": 94, "side": 512}
GREEN, MAGENTA, WHITE = (0.3, 1.0, 0.45), (1.0, 0.25, 0.85), (1.0, 1.0, 1.0)


def now() -> str:
    """Return the current UTC time as ISO text.

    Returns
    -------
    str
        ISO-8601 timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def reserve(root: Path, slug: str) -> Path:
    """Atomically create the next free numbered ``NN_<slug>`` attempt folder.

    Parameters
    ----------
    root : Path
        Folder holding numbered attempts.
    slug : str
        Snake-case attempt name.

    Returns
    -------
    Path
        Newly created, empty attempt folder.
    """
    assert re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", slug), f"Slug must be snake_case: {slug}"
    while True:
        numbers = [int(m.group(1)) for p in root.iterdir() if p.is_dir() and (m := re.match(r"^(\d+)_", p.name))]
        candidate = root / f"{max(numbers, default=-1) + 1:02d}_{slug}"
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue


def frame_list(centers: Path) -> pd.DataFrame:
    """Read the unique frames reviewed by the cell-centers attempt.

    Parameters
    ----------
    centers : Path
        Cell-centers attempt folder.

    Returns
    -------
    pd.DataFrame
        Sorted ``fov``/``t`` rows, exactly 20.
    """
    frames = pd.read_parquet(centers / "center_coverage.parquet", columns=["dataset", "fov", "t"]).drop_duplicates()
    assert set(frames.dataset) == {DATASET}
    frames = frames[["fov", "t"]].sort_values(["fov", "t"]).reset_index(drop=True)
    frames["t"] = frames.t.astype(int)
    assert len(frames) == 20, f"Expected 20 frames, found {len(frames)}"
    return frames


def read_frame(config, source_attempt: Path, fov: str, t: int) -> dict:
    """Read one frame's projections, tracked nuclei, verified tracks, and existing labels.

    Parameters
    ----------
    config : dynaclr.phenotyping.config.Config
        Source attempt configuration.
    source_attempt : Path
        Completed source attempt with native-resolution cell labels.
    fov : str
        Row/well/position identity.
    t : int
        Zero-based frame.

    Returns
    -------
    dict
        Native arrays and per-arm projection windows for the frame.
    """
    source, focus, tracking = (
        read_position(Path(root) / fov) for root in (config.source, config.focus_source, config.tracks)
    )
    validate_alignment(source, focus, tracking)
    assert np.allclose(source.pixel_size_yx_um, (0.1133, 0.1133), rtol=0, atol=1e-9)
    focus_plane = focus.attrs["focus_slice"][config.focus_channel]["per_timepoint"][str(t)]
    five = projection_window(focus_plane, source.shape[2])
    slab1 = slice(five.start, five.start + 3)
    status = json.loads((source_attempt / "frames" / fov / f"t{t:06d}" / "status.json").read_text())
    assert status["state"] == "complete"
    assert (status["focus"], status["z_start"], status["z_stop"]) == (int(focus_plane), five.start, five.stop)
    with (
        closing(zarr.storage.LocalStore(str(source.path), read_only=True)) as source_store,
        closing(zarr.storage.LocalStore(str(focus.path), read_only=True)) as focus_store,
        closing(zarr.storage.LocalStore(str(tracking.path), read_only=True)) as track_store,
        closing(zarr.storage.LocalStore(str(source_attempt / "segmentations.ome.zarr"), read_only=True)) as labels,
    ):
        array = zarr.open_group(store=source_store, mode="r")[source.array_path]
        membrane = np.asarray(array[t, source.channels.index(config.membrane_channel), five])
        phase = np.asarray(array[t, source.channels.index(config.focus_channel), five])
        prepared = np.asarray(
            zarr.open_group(store=focus_store, mode="r")[focus.array_path][
                t, focus.channels.index(config.focus_channel), five
            ]
        )
        nuclei = np.asarray(zarr.open_group(store=track_store, mode="r")[tracking.array_path][t, 0, 0])
        group = zarr.open_group(store=labels, mode="r")[fov]
        existing = np.asarray(group["0"][t, 0, 0])
        existing_nuclei = np.asarray(group["labels/nuclei/0"][t, 0, 0])
    verify_phase_alignment(phase, prepared, config.phase_alignment_rtol, config.phase_alignment_atol)
    assert np.issubdtype(nuclei.dtype, np.integer) and not np.any(nuclei < 0)
    assert existing.dtype == np.uint32 and existing.shape == nuclei.shape
    tracks = validate_tracks(pd.read_csv(find_track_csv(config.tracks, fov)))
    verified = resolve_track_labels(nuclei, tracks[tracks.t == t], config.track_label_mode, config.track_id_stride)
    five_mip = project(membrane)
    return {
        "fov": fov,
        "t": t,
        "focus": int(focus_plane),
        "depth": source.shape[2],
        "pixel_size": source.pixel_size_yx_um,
        "windows": {"existing_058_5plane": five, "native_5plane": five, "native_slab1": slab1},
        "membrane": {
            "existing_058_5plane": five_mip,
            "native_5plane": five_mip,
            "native_slab1": membrane[:3].max(axis=0),
        },
        "phase": project(phase),
        "nuclei": nuclei,
        "nuclei_match_source_attempt": bool(np.array_equal(existing_nuclei, nuclei)),
        "tracks": verified,
        "existing": existing,
    }


def measure(arm: str, frame: dict, cells: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join tracked nuclei to one arm's cells and tabulate per-cell geometry.

    Parameters
    ----------
    arm : str
        Arm name.
    frame : dict
        Output of :func:`read_frame`.
    cells : np.ndarray
        Native integer cell labels.

    Returns
    -------
    tuple of pd.DataFrame
        Per-cell table and per-nucleus membership.
    """
    nuclei = frame["nuclei"]
    membership = map_nuclei_to_cells(cells, nuclei, frame["tracks"])
    ids, area = np.unique(cells[cells > 0], return_counts=True)
    edge = np.unique(np.concatenate([cells[0], cells[-1], cells[:, 0], cells[:, -1]]))
    overlapped = np.unique(cells[(nuclei > 0) & (cells > 0)])
    assigned = membership.loc[membership.assignment_status == "matched", "cell_id"].value_counts()
    table = pd.DataFrame(
        {
            "arm": arm,
            "fov": frame["fov"],
            "t": frame["t"],
            "cell_id": ids.astype(np.int64),
            "area_px": area.astype(np.int64),
        }
    )
    table["area_um2"] = table.area_px * frame["pixel_size"][0] * frame["pixel_size"][1]
    table["touches_border"] = np.isin(ids, edge)
    table["nuclei_in_cell"] = table.cell_id.map(assigned).fillna(0).astype(int)
    table["nuclear_overlap"] = np.isin(ids, overlapped)
    membership["cell_touches_border"] = pd.array(
        membership.cell_id.map(table.set_index("cell_id").touches_border), dtype="boolean"
    )
    assert (membership.t == frame["t"]).all()
    membership.insert(0, "fov", frame["fov"])
    membership.insert(0, "arm", arm)
    return table, membership


def counts(cells: pd.DataFrame, matched: pd.DataFrame, prefix: str = "") -> dict:
    """Count cells, shared masks, nucleus-free cells, and small cells.

    Parameters
    ----------
    cells : pd.DataFrame
        Per-cell rows from :func:`measure`.
    matched : pd.DataFrame
        Matched membership rows belonging to ``cells``.
    prefix : str
        Column-name prefix.

    Returns
    -------
    dict
        Named counts, shared-mask fraction, and median cell area.
    """
    shared = int((matched.nuclei_in_cell > 1).sum())
    return {
        f"{prefix}n_cells": len(cells),
        f"{prefix}matched": len(matched),
        f"{prefix}shared_mask_nuclei": shared,
        f"{prefix}shared_mask_fraction": shared / len(matched) if len(matched) else np.nan,
        f"{prefix}shared_masks": int((cells.nuclei_in_cell > 1).sum()),
        f"{prefix}cells_no_matched_nucleus": int((cells.nuclei_in_cell == 0).sum()),
        f"{prefix}cells_no_nuclear_overlap": int((~cells.nuclear_overlap).sum()),
        f"{prefix}median_area_um2": float(cells.area_um2.median()) if len(cells) else np.nan,
        f"{prefix}cells_below_46p5_um2": int((cells.area_um2 < FLOOR_UM2).sum()),
    }


def metrics(cells: pd.DataFrame, membership: pd.DataFrame) -> dict:
    """Summarize nucleus assignment and cell counts, overall and away from the border.

    Parameters
    ----------
    cells : pd.DataFrame
        Per-cell rows for one arm (one frame or pooled).
    membership : pd.DataFrame
        Matching per-nucleus rows.

    Returns
    -------
    dict
        Track-row, assignment, and cell counts.
    """
    status = membership.assignment_status
    matched = membership[status == "matched"]
    record = {
        "n_track_rows": len(membership),
        "n_tracked_nuclei": int((membership.nucleus_status == "verified").sum()),
        "ambiguous": int((status == "ambiguous").sum()),
        "unmatched": int((status == "unmatched").sum()),
        "missing_nuclear_footprint": int((status == "missing_nuclear_footprint").sum()),
    }
    record.update(counts(cells, matched))
    record.update(
        counts(cells[~cells.touches_border], matched[~matched.cell_touches_border.astype(bool)], "nonborder_")
    )
    return record


def correctness(frames: pd.DataFrame, source_attempt: Path, membership: pd.DataFrame, nuclei_equal: bool) -> dict:
    """Compare recomputed existing-arm assignments to the source attempt's observations.

    Parameters
    ----------
    frames : pd.DataFrame
        Selected ``fov``/``t`` rows.
    source_attempt : Path
        Completed source attempt.
    membership : pd.DataFrame
        Recomputed ``existing_058_5plane`` membership.
    nuclei_equal : bool
        Whether tracked nuclei equal the source attempt's stored nuclear labels.

    Returns
    -------
    dict
        Row counts, per-field equality, and the overall verdict.
    """
    columns = ["dataset", "fov", "t", "track_id", "marker", "assignment_status", "cell_id", "nuclei_in_cell"]
    reference = pd.read_parquet(source_attempt / "observations.parquet", columns=columns)
    reference = reference[reference.marker == MARKER].merge(frames, on=["fov", "t"])
    assert set(reference.dataset) == {DATASET} and not reference.duplicated(KEYS).any()
    mine = membership[KEYS + ["assignment_status", "cell_id", "nuclei_in_cell"]]
    joined = reference.merge(mine, on=KEYS, how="outer", suffixes=("_reference", "_recomputed"), indicator=True)
    cell = [pd.array(joined[f"cell_id_{side}"], dtype="Int64").fillna(-1) for side in ("reference", "recomputed")]
    check = {
        "marker_rows_used": MARKER,
        "frames": len(frames),
        "reference_rows": len(reference),
        "recomputed_rows": len(mine),
        "rows_paired_one_to_one": bool((joined._merge == "both").all()),
        "assignment_status_equal": bool(
            (joined.assignment_status_reference == joined.assignment_status_recomputed).all()
        ),
        "cell_id_equal": bool((cell[0] == cell[1]).all()),
        "nuclei_in_cell_equal": bool((joined.nuclei_in_cell_reference == joined.nuclei_in_cell_recomputed).all()),
        "tracked_nuclei_equal_source_attempt_nuclei": nuclei_equal,
    }
    check["passed"] = all(value for key, value in check.items() if isinstance(value, bool))
    return check


def panel(image: np.ndarray, cells: np.ndarray, nuclear_edges: np.ndarray, shared: np.ndarray, dim: float):
    """Draw integer-label cell boundaries and nuclear boundaries over a stretched image.

    Parameters
    ----------
    image : np.ndarray
        Native 2D image.
    cells : np.ndarray
        Native integer cell labels.
    nuclear_edges : np.ndarray
        Dilated tracked nuclear boundary mask.
    shared : np.ndarray
        Cell IDs holding more than one matched tracked nucleus.
    dim : float
        Gray-level multiplier after the 1-99 percentile stretch.

    Returns
    -------
    np.ndarray
        RGB overlay at native resolution.
    """
    low, high = np.percentile(image, (1, 99))
    gray = dim * np.clip((image - low) / max(float(high - low), 1e-9), 0, 1)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    rgb[binary_dilation(find_boundaries(cells, mode="inner"))] = GREEN
    rgb[binary_dilation(find_boundaries(np.where(np.isin(cells, shared), cells, 0), mode="inner"))] = MAGENTA
    rgb[nuclear_edges] = WHITE
    return rgb


def render(target: Path, frame: dict, results: dict, zoom: dict | None = None) -> str:
    """Write the arm-by-row comparison figure for one frame as PNG and PDF.

    Parameters
    ----------
    target : Path
        Figure folder.
    frame : dict
        Output of :func:`read_frame`.
    results : dict
        Per-arm labels, cell tables, and frame metrics.
    zoom : dict or None
        Track and native box side for a zoomed figure; ``None`` shows the full FOV.

    Returns
    -------
    str
        File stem written.
    """
    fov, t = frame["fov"], frame["t"]
    stem = f"{fov.replace('/', '_')}_t{t:03d}"
    box = None
    if zoom is not None:
        anchor = frame["tracks"].set_index("track_id").loc[zoom["track_id"]]
        box = bounds(anchor.y, anchor.x, zoom["side"])
        stem += f"_track{zoom['track_id']}_zoom{zoom['side']}"
    nuclear_edges = binary_dilation(find_boundaries(frame["nuclei"], mode="inner"))
    fig = Figure(figsize=(16, 11.8) if zoom else (16, 16.2))
    axes = fig.subplots(2, 3)
    for column, arm in enumerate(ARMS):
        result = results[arm]
        shared = result["cells"].loc[result["cells"].nuclei_in_cell > 1, "cell_id"].to_numpy()
        window = frame["windows"][arm]
        stats = (
            f"{result['metrics']['n_cells']} cells; shared-mask nuclei "
            f"{result['metrics']['shared_mask_fraction']:.1%} "
            f"({result['metrics']['shared_mask_nuclei']}/{result['metrics']['matched']})"
        )
        if zoom is not None:
            member = result["membership"].set_index("track_id").loc[zoom["track_id"]]
            cell_id = "none" if pd.isna(member.cell_id) else int(member.cell_id)
            stats += f"\ntrack {zoom['track_id']}: {member.assignment_status}, cell {cell_id}, "
            stats += f"nuclei_in_cell={int(member.nuclei_in_cell)}"
        grid = " (0.58 um grid)" if arm == "existing_058_5plane" else " (native grid)"
        rows = (
            (frame["membrane"][arm], 0.8, f"membrane MIP z[{window.start},{window.stop}){grid}"),
            (frame["phase"], 0.65, "Phase3D 5-plane MIP, dimmed"),
        )
        for row, (image, dim, label) in enumerate(rows):
            rgb = panel(image, result["labels"], nuclear_edges, shared, dim)
            if box is not None:
                rgb = np.stack([padded_crop(rgb[..., c], box) for c in range(3)], axis=-1)
            axis = axes[row, column]
            axis.imshow(rgb, origin="upper")
            if box is not None:
                axis.plot(anchor.x - box[1], anchor.y - box[0], "+", color="yellow", markersize=12, mew=2)
            axis.set_title(f"{arm}: {label}\n{stats}", fontsize=10)
            axis.axis("off")
    area = "512-native-px box on track 94" if zoom else "full FOV"
    fig.suptitle(
        f"{DATASET} {fov} t{t}, Phase3D focus z={frame['focus']}; {area}\n"
        "Green: cell boundaries from integer labels. Magenta: cells holding >1 matched tracked nucleus. "
        "White: tracked nuclei.\nDisplay 1-99 percentile stretch per image. No ground-truth membrane; "
        "no size filter applied.",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88 if zoom else 0.91, bottom=0.01, wspace=0.03, hspace=0.14)
    fig.savefig(target / f"{stem}.png", dpi=200)
    fig.savefig(target / f"{stem}.pdf", dpi=200)
    return stem


def git(repo: Path, *args: str) -> str:
    """Run a read-only Git query in the repository.

    Parameters
    ----------
    repo : Path
        Repository root.
    *args : str
        Git arguments.

    Returns
    -------
    str
        Standard output.
    """
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def run(source_attempt: Path, centers_attempt: Path, slug: str, target: Path) -> dict:
    """Run all arms on the selected frames and write tables, labels, figures, and provenance.

    Parameters
    ----------
    source_attempt : Path
        Completed attempt with 0.58 um labels and ``pilot.yml``.
    centers_attempt : Path
        Cell-centers attempt defining the frames.
    slug : str
        Attempt slug.
    target : Path
        Reserved, empty attempt folder.

    Returns
    -------
    dict
        Final summary.
    """
    start = time.perf_counter()
    assert_compute_node()
    root = source_attempt.parent
    assert centers_attempt.parent == root and target.parent == root
    assert re.fullmatch(rf"\d+_{slug}", target.name) and target.is_dir()
    assert {p.name for p in target.iterdir()} <= {"run.log"}, f"{target} is not a fresh attempt"
    print(f"ATTEMPT {target} on {platform.node()} at {now()}", flush=True)
    config = load_config(source_attempt / "pilot.yml")
    manifest = json.loads((source_attempt / "manifest.json").read_text())
    assert config.dataset == DATASET and Path(config.output) == source_attempt
    assert json.loads(json.dumps(asdict(config.segmentation))) == manifest["scientific"]["config"]["segmentation"]
    segmentation = replace(config.segmentation, target_pixel_size_um=None)
    assert "diameter=None" in inspect.getsource(CPDinoSegmenter.__call__)
    frames = frame_list(centers_attempt)
    repo = Path(__file__).resolve().parents[3]
    inputs = [
        source_attempt / "pilot.yml",
        source_attempt / "manifest.json",
        source_attempt / "observations.parquet",
        centers_attempt / "center_coverage.parquet",
    ]
    hashes = {str(p): file_sha(p) for p in inputs}
    provenance = target / "provenance"
    provenance.mkdir()
    helper = Path(__file__).with_name("review_cytoplasm_coverage.py")
    shutil.copy2(__file__, provenance / Path(__file__).name)
    shutil.copy2(helper, provenance / helper.name)
    (provenance / "command.txt").write_text(
        f"host: {platform.node()}\ncwd: {os.getcwd()}\n"
        f"reserved by: {shlex.join([sys.executable, __file__])} --source-attempt {source_attempt} "
        f"--centers-attempt {centers_attempt} --slug {slug} --reserve-only (atomic mkdir)\n"
        f"command: {shlex.join([sys.executable, *sys.argv])} 2>&1 | tee {target / 'run.log'}\n"
    )
    atomic_json(provenance / "segmentation.json", asdict(segmentation))
    atomic_json(
        provenance / "provenance.json",
        {
            "started_at": now(),
            "source_attempt": str(source_attempt),
            "centers_attempt": str(centers_attempt),
            "slug": slug,
            "target": str(target),
            "argv": sys.argv,
            "hostname": platform.node(),
            "python": sys.executable,
            "script_sha256": file_sha(Path(__file__)),
            "helper_sha256": {helper.name: file_sha(helper)},
            "input_sha256": hashes,
            "git_base": git(repo, "rev-parse", "HEAD"),
            "git_branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "code_identity": code_identity(),
            "source_segmentation": asdict(config.segmentation),
            "segmentation": asdict(segmentation),
            "segmenter": f"{CPDinoSegmenter.__module__}.{CPDinoSegmenter.__name__}",
            "segmenter_eval": "cellpose CellposeModel.eval, normalize=True (1-99 pct), diameter=None, bsize default",
            "arms": ARM_DEFINITIONS,
            "frames": frames.to_dict("records"),
            "cell_area_floor_um2_reported_not_applied": FLOOR_UM2,
            "border_rule": "cell has a pixel in the first/last row or column (bbox reaches the image edge)",
        },
    )
    loaded = {}
    results = {}
    for row in frames.itertuples():
        frame = read_frame(config, source_attempt, row.fov, int(row.t))
        loaded[(row.fov, int(row.t))] = frame
        cells, membership = measure("existing_058_5plane", frame, frame["existing"])
        results[(row.fov, int(row.t), "existing_058_5plane")] = {
            "labels": frame["existing"],
            "cells": cells,
            "membership": membership,
        }
        print(f"READ {row.fov} t{row.t}: focus {frame['focus']}, {len(frame['tracks'])} track rows", flush=True)
    existing = pd.concat(
        [value["membership"] for key, value in results.items() if key[2] == "existing_058_5plane"], ignore_index=True
    )
    check = correctness(
        frames, source_attempt, existing, all(f["nuclei_match_source_attempt"] for f in loaded.values())
    )
    print(f"CORRECTNESS {json.dumps(check)}", flush=True)
    atomic_json(target / "summary.json", {"complete": False, "correctness_check": check})
    if not check["passed"]:
        raise RuntimeError("Existing-arm recomputation differs from the source attempt; stopping before inference")
    segmenter = CPDinoSegmenter(segmentation)
    net = segmenter.model.net
    reference_model = json.loads((source_attempt / "provenance" / "model.json").read_text())
    model = {
        "identity": segmenter.identity,
        "same_checkpoint_as_source_attempt": segmenter.identity == reference_model,
        "network_dtype": str(net.dtype),
        "parameter_dtypes": sorted({str(p.dtype) for p in net.parameters()}),
        "device": str(segmenter.model.device),
        "diameter": None,
        "normalize": "cellpose normalize=True (1, 99) percentiles",
    }
    atomic_json(provenance / "model.json", model)
    print(f"MODEL {json.dumps(model)}", flush=True)
    (target / "labels").mkdir()
    timings = []
    for (fov, t), frame in loaded.items():
        for arm in NATIVE_ARMS:
            before = time.perf_counter()
            labels = segmenter(frame["membrane"][arm], frame["pixel_size"])
            seconds = time.perf_counter() - before
            assert labels.shape == frame["nuclei"].shape and labels.dtype == np.uint32
            window = frame["windows"][arm]
            np.savez_compressed(
                target / "labels" / f"{fov.replace('/', '_')}_t{t:03d}_{arm}.npz",
                labels=labels,
                z_start=window.start,
                z_stop=window.stop,
            )
            cells, membership = measure(arm, frame, labels)
            results[(fov, t, arm)] = {"labels": labels, "cells": cells, "membership": membership}
            timings.append({"fov": fov, "t": t, "arm": arm, "segmentation_seconds": seconds})
            print(f"SEGMENTED {fov} t{t} {arm}: {len(cells)} cells in {seconds:.1f}s", flush=True)
    records = []
    for (fov, t, arm), result in results.items():
        frame = loaded[(fov, t)]
        window = frame["windows"][arm]
        result["metrics"] = metrics(result["cells"], result["membership"])
        records.append(
            {
                "arm": arm,
                "fov": fov,
                "t": t,
                "focus": frame["focus"],
                "z_start": window.start,
                "z_stop": window.stop,
                "n_planes": window.stop - window.start,
                **result["metrics"],
            }
        )
    order = {arm: index for index, arm in enumerate(ARMS)}
    table = pd.DataFrame(records).sort_values(["arm", "fov", "t"], key=lambda s: s.map(order) if s.name == "arm" else s)
    table.to_csv(target / "recipe_metrics.csv", index=False)
    all_cells = pd.concat([r["cells"] for r in results.values()], ignore_index=True)
    all_membership = pd.concat([r["membership"] for r in results.values()], ignore_index=True)
    atomic_parquet(target / "recipe_cells.parquet", all_cells)
    atomic_parquet(target / "recipe_membership.parquet", all_membership)
    pd.DataFrame(timings).to_csv(target / "segmentation_timings.csv", index=False)
    pooled = []
    for arm in ARMS:
        record = {"arm": arm, "frames": len(frames)}
        record.update(metrics(all_cells[all_cells.arm == arm], all_membership[all_membership.arm == arm]))
        pooled.append(record)
    summary = pd.DataFrame(pooled)
    summary.to_csv(target / "recipe_summary.csv", index=False)
    fractions = table.pivot_table(
        index=["fov", "t"], columns="arm", values="shared_mask_fraction", aggfunc="first", dropna=False
    )[list(ARMS)]
    print(f"POOLED\n{summary.to_string(index=False)}\nSHARED FRACTION BY FRAME\n{fractions.to_string()}", flush=True)
    figures = target / "figures"
    figures.mkdir()
    pages = []
    for (fov, t), frame in loaded.items():
        arms = {arm: results[(fov, t, arm)] for arm in ARMS}
        pages.append(render(figures, frame, arms))
        if (fov, t) == (ZOOM["fov"], ZOOM["t"]):
            pages.append(render(figures, frame, arms, ZOOM))
        print(f"RENDERED {pages[-1]}", flush=True)
    assert all(file_sha(Path(p)) == digest for p, digest in hashes.items()), "Source inputs changed"
    report = {
        "complete": True,
        "attempt": str(target),
        "finished_at": now(),
        "runtime_seconds": time.perf_counter() - start,
        "segmentation_seconds_total": float(sum(t["segmentation_seconds"] for t in timings)),
        "correctness_check": check,
        "model": model,
        "arms": ARM_DEFINITIONS,
        "pooled": json.loads(summary.to_json(orient="records")),
        "shared_mask_fraction_by_frame": json.loads(fractions.reset_index().to_json(orient="records")),
        "figures": pages,
        "source_inputs_unchanged": True,
        "note": "Shared masks are matched tracked nuclei whose cell holds >1 matched nucleus. Cells below "
        f"{FLOOR_UM2} um2 are counted, not removed. No ground-truth membrane; differences are not accuracy.",
    }
    atomic_json(target / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    """Parse the command line and reserve or run one attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-attempt", type=Path, required=True)
    parser.add_argument("--centers-attempt", type=Path, required=True)
    parser.add_argument("--slug", required=True, help="snake_case name for the new NN_<slug> attempt folder")
    parser.add_argument("--target", type=Path, default=None, help="Folder printed by --reserve-only")
    parser.add_argument("--reserve-only", action="store_true", help="Reserve the attempt folder, print it, and exit")
    args = parser.parse_args()
    source, centers = args.source_attempt.resolve(), args.centers_attempt.resolve()
    if args.reserve_only:
        print(reserve(source.parent, args.slug))
        return
    target = args.target.resolve() if args.target else reserve(source.parent, args.slug)
    run(source, centers, args.slug, target)


if __name__ == "__main__":
    main()
