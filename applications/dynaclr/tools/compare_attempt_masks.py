"""Render reference-versus-pilot cell masks for every completed pilot frame of two phenotyping attempts.

Each attempt is a ``dynaclr.phenotyping.pipeline`` output folder. Display images are recomputed from each
attempt's own ``pilot.yml`` and ``status.parquet`` row: the source membrane channel max-projected over
``focus-k .. focus+k`` when ``segmentation.slab_halfwidth`` is ``k``, else over the status ``z_start .. z_stop``
five-plane window, and the Phase3D five-plane MIP. Counts come from the stored cell and nuclear labels and are
checked against the attempt's ``membership.parquet``.

Usage::

    python compare_attempt_masks.py --pilot NEW --reference OLD --out DIR [--frame A/2/0000:30] [--max-frames 2]
"""

from pathlib import Path

import click
import numpy as np
import pandas as pd
import yaml
from iohub import open_ome_zarr
from matplotlib.figure import Figure
from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries

from dynaclr.phenotyping.io import assert_compute_node

FLOOR_UM2 = 46.5
STATUSES = ("matched", "ambiguous", "unmatched", "missing_nuclear_footprint")
COUNTS = (
    "n_cells",
    "tracked_total",
    *(f"tracked_{s}" for s in STATUSES),
    "tracked_without_cell",
    "nuclear_footprints",
    "assigned_footprints",
    "shared_masks",
    "nuclei_in_shared_masks",
    "cells_below_floor",
)
GREEN, MAGENTA, WHITE = (0.3, 1.0, 0.45), (1.0, 0.25, 0.85), (1.0, 1.0, 1.0)


def frame_counts(
    cells: np.ndarray, nuclei: np.ndarray, tracked: np.ndarray, pixel_area_um2: float
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Assign nuclear footprints to cells by strict majority and count shared and unassigned nuclei.

    Parameters
    ----------
    cells : np.ndarray
        2D integer cell labels, 0 background.
    nuclei : np.ndarray
        2D integer nuclear labels on the same grid, 0 background.
    tracked : np.ndarray
        Nuclear label of each tracked-nucleus row for the frame; 0 marks a row without a label.
    pixel_area_um2 : float
        Area of one pixel in square micrometers.

    Returns
    -------
    counts : dict
        Cell, footprint, shared-mask and tracked-nucleus counts plus median cell area.
    shared : np.ndarray
        Cell IDs holding at least two nuclear footprints.
    areas : np.ndarray
        Area in square micrometers of every cell.
    """
    ids, pixels = np.unique(cells[cells > 0], return_counts=True)
    areas = pixels * pixel_area_um2
    inside = nuclei > 0
    pairs = pd.DataFrame({"nucleus": nuclei[inside], "cell": cells[inside]})
    size = pairs.nucleus.value_counts()
    overlap = pairs[pairs.cell > 0].groupby(["nucleus", "cell"]).size().rename("pixels").reset_index()
    owners = overlap[2 * overlap.pixels > overlap.nucleus.map(size)].set_index("nucleus").cell
    per_cell = owners.value_counts()
    shared = per_cell.index[per_cell >= 2].to_numpy()
    status = np.select(
        [np.isin(tracked, owners.index), np.isin(tracked, overlap.nucleus), np.isin(tracked, size.index)],
        STATUSES[:3],
        STATUSES[3],
    )
    counts = {
        "n_cells": len(ids),
        "tracked_total": len(tracked),
        **{f"tracked_{s}": int((status == s).sum()) for s in STATUSES},
    }
    counts["tracked_without_cell"] = counts["tracked_ambiguous"] + counts["tracked_unmatched"]
    counts.update(
        nuclear_footprints=len(size),
        assigned_footprints=len(owners),
        shared_masks=len(shared),
        nuclei_in_shared_masks=int(per_cell[per_cell >= 2].sum()),
        cells_below_floor=int((areas < FLOOR_UM2).sum()),
        median_cell_area_um2=float(np.median(areas)) if len(areas) else np.nan,
    )
    counts["shared_fraction"] = (
        counts["nuclei_in_shared_masks"] / counts["assigned_footprints"] if len(owners) else np.nan
    )
    return counts, shared, areas


def read_recipe(attempt: Path) -> dict:
    """Read an attempt's recipe keys, frame status and tracked-nucleus membership.

    Parameters
    ----------
    attempt : Path
        Pipeline attempt folder.

    Returns
    -------
    dict
        Source path, channel names, slab halfwidth, status indexed by ``fov``/``t``, and membership.
    """
    raw = yaml.safe_load((attempt / "pilot.yml").read_text())
    status = pd.read_parquet(attempt / "status.parquet").set_index(["fov", "t"])
    assert status.index.is_unique, f"{attempt}: duplicate fov/t rows in status.parquet"
    return {
        "attempt": attempt,
        "dataset": raw["dataset"],
        "source": attempt / raw["source"],
        "membrane": raw["channels"]["membrane"],
        "phase": raw["channels"]["focus"],
        "halfwidth": raw["segmentation"].get("slab_halfwidth"),
        "status": status,
        "membership": pd.read_parquet(
            attempt / "membership.parquet", columns=["fov", "t", "nucleus_label", "assignment_status"]
        ),
    }


def read_arm(recipe: dict, fov: str, t: int) -> dict:
    """Read one attempt's display images, labels and counts for a frame.

    Parameters
    ----------
    recipe : dict
        Output of :func:`read_recipe`.
    fov : str
        Row/well/position identity.
    t : int
        Zero-based frame.

    Returns
    -------
    dict
        ``state`` and ``skip_reason``; for complete frames also images, labels, window, counts and areas.
    """
    if (fov, t) not in recipe["status"].index:
        return {"state": "not_run", "skip_reason": None}
    row = recipe["status"].loc[(fov, t)]
    if row.state != "complete":
        return {"state": row.state, "skip_reason": row.skip_reason}
    focus = int(row.focus)
    five = slice(focus - 1, focus + 4)
    halfwidth = recipe["halfwidth"]
    if halfwidth is None:
        assert (row.z_start, row.z_stop) == (five.start, five.stop), f"{fov} t{t}: status window differs"
        window = five
    else:
        window = slice(focus - halfwidth, focus + halfwidth + 1)
    with open_ome_zarr(recipe["source"] / fov, mode="r", layout="fov") as source:
        names = source.channel_names
        assert 0 <= min(window.start, five.start) and max(window.stop, five.stop) <= source.data.shape[2]
        membrane = np.asarray(source.data[t, names.index(recipe["membrane"]), window]).max(axis=0)
        phase = np.asarray(source.data[t, names.index(recipe["phase"]), five]).max(axis=0)
    with open_ome_zarr(recipe["attempt"] / "segmentations.ome.zarr" / fov, mode="r", layout="fov") as store:
        assert all(axis.unit == "micrometer" for axis in store.axes[-2:])
        pixel_area = store.scale[-2] * store.scale[-1]
        cells = np.asarray(store.data[t, 0, 0])
        nuclei = np.asarray(store.zgroup["labels/nuclei/0"][t, 0, 0])
    members = recipe["membership"]
    members = members[(members.fov == fov) & (members.t == t)]
    tracked = members.nucleus_label.fillna(0).to_numpy(dtype=np.int64)
    counts, shared, areas = frame_counts(cells, nuclei, tracked, pixel_area)
    expected = members.assignment_status.value_counts()
    assert set(expected.index) <= set(STATUSES), f"{fov} t{t}: unknown statuses {set(expected.index)}"
    recount = {s: counts[f"tracked_{s}"] for s in STATUSES}
    assert recount == {s: int(expected.get(s, 0)) for s in STATUSES}, (
        f"{recipe['attempt']} {fov} t{t}: label recount {recount} differs from membership.parquet {expected.to_dict()}"
    )
    return {
        "state": "complete",
        "skip_reason": None,
        "focus": focus,
        "window": window,
        "five": five,
        "membrane": membrane,
        "phase": phase,
        "cells": cells,
        "tracked_edges": binary_dilation(find_boundaries(np.where(np.isin(nuclei, tracked), nuclei, 0), mode="inner")),
        "shared": shared,
        "counts": counts,
        "areas": areas,
    }


def panel(image: np.ndarray, cells: np.ndarray, nuclear_edges: np.ndarray, shared: np.ndarray, dim: float):
    """Draw cell, shared-cell and tracked-nucleus boundaries over a 1-99 percentile stretched image.

    Parameters
    ----------
    image : np.ndarray
        Native 2D image.
    cells : np.ndarray
        Native integer cell labels.
    nuclear_edges : np.ndarray
        Dilated tracked nuclear boundary mask.
    shared : np.ndarray
        Cell IDs holding at least two nuclear footprints.
    dim : float
        Gray-level multiplier after the stretch.

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


def stats(counts: dict) -> str:
    """Format one arm's counts for a panel title.

    Parameters
    ----------
    counts : dict
        Output of :func:`frame_counts`.

    Returns
    -------
    str
        Three title lines.
    """
    return (
        f"{counts['n_cells']} cells; tracked nuclei matched {counts['tracked_matched']}/{counts['tracked_total']}\n"
        f"nuclei in shared masks {counts['nuclei_in_shared_masks']} ({counts['shared_fraction']:.1%}); "
        f"tracked without cell {counts['tracked_without_cell']}\n"
        f"cells < {FLOOR_UM2} µm²: {counts['cells_below_floor']}"
    )


def render(out: Path, dataset: str, fov: str, t: int, arms: list[tuple[str, dict]]) -> Path:
    """Write the two-row, reference-then-pilot comparison figure as PNG and PDF.

    Parameters
    ----------
    out : Path
        Figure folder.
    dataset : str
        Dataset name.
    fov : str
        Row/well/position identity.
    t : int
        Zero-based frame.
    arms : list of tuple
        ``(label, arm)`` pairs from :func:`read_arm`, reference first; the last arm is complete.

    Returns
    -------
    Path
        PNG written.
    """
    pilot = arms[-1][1]
    fig = Figure(figsize=(12, 19.6))
    axes = fig.subplots(2, 2)
    for column, (label, arm) in enumerate(arms):
        if arm["state"] != "complete":
            note = "not run" if arm["state"] == "not_run" else f"{arm['state']}: {arm['skip_reason']}"
            for row in range(2):
                axis = axes[row, column]
                axis.imshow(np.full(pilot["cells"].shape, 0.5), cmap="gray", vmin=0, vmax=1)
                axis.text(0.5, 0.5, f"reference frame {note}", transform=axis.transAxes, ha="center")
                axis.set_title(label, fontsize=10)
                axis.axis("off")
            continue
        window, five = arm["window"], arm["five"]
        rows = (
            (arm["membrane"], 0.8, f"VS membrane MIP z[{window.start},{window.stop})"),
            (arm["phase"], 0.65, f"Phase3D 5-plane MIP z[{five.start},{five.stop}), dimmed"),
        )
        for row, (image, dim, name) in enumerate(rows):
            axis = axes[row, column]
            axis.imshow(panel(image, arm["cells"], arm["tracked_edges"], arm["shared"], dim), origin="upper")
            axis.set_title(f"{label}: {name}\n{stats(arm['counts'])}", fontsize=10)
            axis.axis("off")
    fig.suptitle(
        f"{dataset} {fov} t{t}, Phase3D focus z={pilot['focus']}\n"
        "Green: cell boundaries. Magenta: cells holding ≥2 nuclear footprints (strict-majority overlap). "
        "White: tracked nuclei.\nDisplay 1-99 percentile stretch per image.",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.915, bottom=0.01, wspace=0.03, hspace=0.12)
    stem = out / f"{fov.replace('/', '_')}_t{t:03d}"
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"), dpi=200)
    return stem.with_suffix(".png")


def summarize(metrics: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Pool counts per arm over all complete frames and over frames complete in both arms.

    Parameters
    ----------
    metrics : pd.DataFrame
        Per frame and arm rows.
    cells : pd.DataFrame
        Per-cell ``role``/``fov``/``t``/``area_um2`` rows.

    Returns
    -------
    pd.DataFrame
        One row per ``scope`` and arm.
    """
    done = metrics[metrics.state == "complete"]
    paired = done.groupby(["fov", "t"]).role.transform("size") == 2
    records = []
    for scope, table in (("all_complete", done), ("paired", done[paired])):
        for role in ("reference", "pilot"):
            group = table[table.role == role]
            areas = cells[cells.role == role].merge(group[["fov", "t"]])
            label = metrics.label[metrics.role == role].iloc[0]
            record = {"scope": scope, "role": role, "label": label, "frames": len(group)}
            record.update(group[list(COUNTS)].sum().astype(int).to_dict())
            assigned = record["assigned_footprints"]
            record["shared_fraction"] = record["nuclei_in_shared_masks"] / assigned if assigned else np.nan
            record["median_cell_area_um2"] = areas.area_um2.median()
            records.append(record)
    return pd.DataFrame(records)


@click.command()
@click.option("--pilot", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--reference", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--pilot-label", default="native DynaCell", show_default=True)
@click.option("--reference-label", default="0.58 µm", show_default=True)
@click.option("--frame", "frames", multiple=True, help="Restrict to FOV:T, e.g. B/2/0000:30; repeatable.")
@click.option("--max-frames", type=int, default=None, help="Render only the first N selected frames.")
def main(
    pilot: Path,
    reference: Path,
    out: Path,
    pilot_label: str,
    reference_label: str,
    frames: tuple[str, ...],
    max_frames: int | None,
) -> None:
    """Compare reference and pilot attempt masks on every completed pilot frame."""
    assert_compute_node()
    recipes = {"reference": read_recipe(reference.resolve()), "pilot": read_recipe(pilot.resolve())}
    labels = {"reference": reference_label, "pilot": pilot_label}
    dataset = recipes["pilot"]["dataset"]
    assert recipes["reference"]["dataset"] == dataset, "Attempts belong to different datasets"
    status = recipes["pilot"]["status"]
    selected = sorted(status.index[status.state == "complete"])
    if frames:
        wanted = [(fov, int(t)) for fov, t in (f.rsplit(":", 1) for f in frames)]
        missing = set(wanted) - set(selected)
        assert not missing, f"Requested frames not complete in pilot: {sorted(missing)}"
        selected = sorted(wanted)
    selected = selected[:max_frames]
    out.mkdir(parents=True, exist_ok=True)
    records, cells = [], []
    for fov, t in selected:
        arms = []
        for role in ("reference", "pilot"):
            arm = read_arm(recipes[role], fov, t)
            arms.append((labels[role], arm))
            record = {
                "role": role,
                "label": labels[role],
                "attempt": str(recipes[role]["attempt"]),
                "dataset": dataset,
                "fov": fov,
                "t": t,
                "state": arm["state"],
                "skip_reason": arm["skip_reason"],
            }
            if arm["state"] == "complete":
                record.update(
                    focus=arm["focus"], membrane_z_start=arm["window"].start, membrane_z_stop=arm["window"].stop
                )
                record.update(arm["counts"])
                cells.append(pd.DataFrame({"role": role, "fov": fov, "t": t, "area_um2": arm["areas"]}))
            records.append(record)
        png = render(out, dataset, fov, t, arms)
        click.echo(f"{png}: " + "; ".join(f"{r['label']} {r['state']}" for r in records[-2:]))
    metrics = pd.DataFrame(records)
    metrics.to_csv(out / "frame_metrics.csv", index=False)
    summary = summarize(metrics, pd.concat(cells, ignore_index=True))
    summary.to_csv(out / "summary.csv", index=False)
    click.echo(summary.to_string(index=False))


if __name__ == "__main__":
    main()
