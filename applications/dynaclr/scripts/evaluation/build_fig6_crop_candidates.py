"""Figure 6 crop-candidate review gallery drawn from the attempt-12 joined table.

Six categories of Phase/fluorescence-agreeing exemplars, 10 random candidates
each under diversity constraints, crop QC with replacement draws, and
Phase | pAL | TOMM20 | merge strips at fixed per-dataset contrast. No fitting.
"""

import json
import shutil
from collections import Counter
from pathlib import Path

import build_fig6_phase_prototype as prototype
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from anndata.io import read_elem
from build_fig6_phase_labels import FIG
from build_fig6_phase_labels import OUTPUT as ATTEMPT12
from build_fig6_phase_prototype import BASE, COLORS, DATES, STATES, SUF, sha, write_json
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

OUTPUT = FIG / "13_crop_candidate_gallery"
TABLE = ATTEMPT12 / "source_data/observation_predictions.parquet"
DISPLAY = ATTEMPT12 / "source_data/display_rows.parquet"
IMAGES = Path("/hpc/projects/organelle_phenotyping/datasets")
PHASE, PAL, TOMM20 = "Phase3D", "raw mCherry EX561 EM600-37", "raw GFP EX488 EM525-45"
PAL_MARKERS = {"04_10_TOMM20": "pAL17", "04_30_TOMM20": "pAL40", "07_01_SEC61B_TOMM20_G3BP1": "pAL17"}
DATASETS = sorted(DATES)
CHANNELS = ["phase", "pal", "tomm20"]
ZARR_LABELS = (TOMM20, PAL, PHASE)
CROP_CHANNELS = dict(phase=PHASE, tomm20=TOMM20, pal=PAL)
REFERENCE_COLUMNS = ["dataset", "perturbation", "pal_state", "fov_name", "track_id", "t", "id"]
HALF = 112
CENTER = slice(56, 168)
SEED = 17
PER_CATEGORY = 10
PER_FOV = 2
REFERENCE_CELLS = 300
INFECTED_PAL, UNINFECTED_PAL = 0.8, 0.2
REMODELED, NOT_REMODELED = 0.8, 0.2
CONFIDENCE, RELAXED_CONFIDENCE = 0.8, 0.6
CATEGORIES = ["INFREM", "UNINF"] + ["INF_" + s for s in STATES]
TITLES = dict(INFREM="infected + remodeled", UNINF="uninfected + not remodeled")
TITLES.update({"INF_" + s: f"infected, PCNA→Phase cycle state {s}" for s in STATES})
CATEGORY_COLORS = dict(INFREM="#d62728", UNINF="#1a1a1a", **{"INF_" + s: c for s, c in zip(STATES, COLORS)})
CMAPS = dict(
    phase="gray",
    pal=LinearSegmentedColormap.from_list("pal", ["black", "#ff00ff"]),
    tomm20=LinearSegmentedColormap.from_list("tomm20", ["black", "#00ff66"]),
)
MAGENTA = np.array([1.0, 0.0, 1.0])
GREEN = np.array([0.0, 1.0, 0.4])
HALO = [pe.withStroke(linewidth=2, foreground="white")]
DARK = [pe.withStroke(linewidth=1.5, foreground="black")]
INFECTED_RULE = (
    "infected = ZIKV/DENV well AND pAL Phase score ≥ 0.8 AND pAL fluorescence state positive (posterior ≥ 0.9)"
)
UNINFECTED_RULE = (
    "uninfected = uninfected well AND pAL Phase score ≤ 0.2 AND pAL fluorescence state negative (posterior ≤ 0.1)"
)
COLUMNS = [
    "dataset",
    "experiment",
    "perturbation",
    "hours_post_perturbation",
    "fov_name",
    "track_id",
    "t",
    "id",
    "x",
    "y",
    "Phase3D_umap1",
    "Phase3D_umap2",
    "pal_phase_score",
    "pal_posterior",
    "pal_state",
    "sec61b_phase_score",
    "cycle_state",
    "cycle_confidence",
]
TILE, GAP, COL_GAP, MARGIN, CAPTION, HEADER, FOOTER = 1.85, 0.04, 0.4, 0.3, 0.5, 1.1, 0.85


def confirm_channels(d, markers=None, experiments=DATES):
    """Assert each marker's embedding zarr reads the expected image channel in every cohort well.

    Parameters
    ----------
    d : pandas.DataFrame
        Cohort table with ``dataset`` and ``fov_name``.
    markers : dict, optional
        ``{dataset: [(marker, zarr channel), ...]}``; defaults to mCherry = pAL and
        GFP = TOMM20 for every attempt-13 dataset.
    experiments : dict, optional
        Dataset key -> experiment directory name.

    Returns
    -------
    dict
        Per-dataset embedding-zarr ``uns`` channel/marker and well coverage.
    """
    if markers is None:
        markers = {ds: [(PAL_MARKERS[ds], PAL), ("TOMM20", TOMM20)] for ds in DATASETS}
    result = {}
    for ds, pairs in markers.items():
        cohort = sorted(set(d.fov_name[d.dataset.eq(ds)].str.rsplit("/", n=1).str[0]))
        result[ds] = dict(cohort_wells=cohort)
        for marker, channel in pairs:
            g = zarr.open_group(str(BASE / experiments[ds] / SUF / "embeddings" / f"{marker}.zarr"), mode="r")
            uns = {k: str(read_elem(g["uns"][k])) for k in ("channel", "marker")}
            assert uns == dict(channel=channel, marker=marker), (ds, marker, uns)
            wells = sorted(set(read_elem(g["obs"]).fov_name.astype(str).str.rsplit("/", n=1).str[0]))
            assert set(cohort) <= set(wells), (ds, marker, wells)
            result[ds][marker] = dict(uns, embedding_wells=wells)
    return result


def position(stores, cache, dataset, fov, labels=None):
    """Return the cached image array, channel indices, focus table and pixel size of one FOV.

    Parameters
    ----------
    stores : dict
        Read-only image zarr groups keyed by dataset.
    cache : dict
        Position cache keyed by ``(dataset, fov_name)``.
    dataset, fov : str
        Dataset key and FOV name.
    labels : dict, optional
        ``{dataset: expected omero channel labels}``; defaults to ``ZARR_LABELS`` for every dataset.

    Returns
    -------
    dict
        ``array``, ``channels``, ``focus`` and ``pixel_um`` for the FOV.
    """
    key = (dataset, fov)
    if key not in cache:
        expected = list(ZARR_LABELS if labels is None else labels[dataset])
        pos = stores[dataset][fov]
        attrs = dict(pos.attrs)
        found = [c["label"] for c in attrs["ome"]["omero"]["channels"]]
        assert found == expected, (dataset, fov, found)
        cache[key] = dict(
            array=pos["0"],
            channels={name: found.index(name) for name in expected},
            focus=attrs.get("focus_slice", {}).get("Phase3D", {}).get("per_timepoint", {}),
            pixel_um=attrs["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][-1],
        )
    return cache[key]


def read_crop(stores, cache, r, channels=CROP_CHANNELS, labels=None):
    """Read one 224x224 crop with the attempt-12 ``crops`` recipe.

    Parameters
    ----------
    stores : dict
        Read-only image zarr groups keyed by dataset.
    cache : dict
        Position cache.
    r : pandas.Series
        One cohort row.
    channels : dict, optional
        Crop key -> zarr channel name; ``phase`` is the focus slice, every other
        key a 16-slice MIP with a ``{key}_center_p99`` QC metric.
    labels : dict, optional
        Expected omero channel labels per dataset (see ``position``).

    Returns
    -------
    tuple of (str or None, dict or None)
        Geometry rejection reason (``None`` if the crop is valid) and, when valid,
        raw ``phase`` focus slice, 16-slice fluorescence MIPs and QC metrics.
    """
    p = position(stores, cache, r.dataset, r.fov_name, labels)
    ar = p["array"]
    t, y, x = int(r.t), int(round(r.y)), int(round(r.x))
    if min(y, x) < HALF or y + HALF > ar.shape[-2] or x + HALF > ar.shape[-1]:
        return "edge", None
    focus = p["focus"].get(str(t))
    if focus is None:
        return "focus_missing", None
    focus = int(focus)
    if focus < 5 or focus + 11 > ar.shape[2]:
        return "focus_out_of_range", None
    ys, xs = slice(y - HALF, y + HALF), slice(x - HALF, x + HALF)
    c = p["channels"]
    crop = {
        key: np.asarray(ar[t, c[name], focus, ys, xs])
        if key == "phase"
        else np.asarray(ar[t, c[name], focus - 5 : focus + 11, ys, xs]).max(0)
        for key, name in channels.items()
    }
    meta = dict(
        z_focus=focus,
        z_size=int(ar.shape[2]),
        edge_margin_px=int(min(y - HALF, x - HALF, ar.shape[-2] - y - HALF, ar.shape[-1] - x - HALF)),
        pixel_um=float(p["pixel_um"]),
    )
    meta.update(
        {f"{key}_center_p99": float(np.percentile(crop[key][CENTER, CENTER], 99)) for key in channels if key != "phase"}
    )
    return None, dict(crop=crop, meta=meta)


def reference_thresholds(
    d,
    stores,
    cache,
    units=DATASETS,
    unit="dataset",
    columns=REFERENCE_COLUMNS,
    channels=CROP_CHANNELS,
    labels=None,
    mixes=("perturbation", "pal_state"),
):
    """Derive per-unit QC thresholds from 300 random geometrically valid cells.

    Parameters
    ----------
    d : pandas.DataFrame
        Cohort table.
    stores : dict
        Read-only image zarr groups keyed by dataset.
    cache : dict
        Position cache.
    units : list of str, optional
        Threshold units in draw order (attempt 13: the datasets).
    unit : str, optional
        Column of ``d`` holding the unit.
    columns : list of str, optional
        Row columns copied into the reference sample.
    channels : dict, optional
        Crop channels (see ``read_crop``); each non-phase key gets a threshold.
    labels : dict, optional
        Expected omero channel labels per dataset.
    mixes : tuple of str, optional
        Columns whose value counts in the sample are reported.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Thresholds (median central-112 p99 per fluorescence channel) with sample
        summaries, and the per-cell reference metrics.
    """
    rng = np.random.default_rng(SEED)
    fluorescence = [key for key in channels if key != "phase"]
    thresholds = {}
    rows = []
    for u in units:
        g = d[d[unit].eq(u)]
        values = []
        skipped = Counter()
        for i in rng.permutation(g.index.to_numpy()):
            r = d.loc[i]
            reason, out = read_crop(stores, cache, r, channels, labels)
            if reason:
                skipped[reason] += 1
                continue
            values.append(dict(r[columns], **out["meta"]))
            if len(values) == REFERENCE_CELLS:
                break
        v = pd.DataFrame(values)
        rows.append(v)
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        thresholds[u] = {key: float(v[f"{key}_center_p99"].median()) for key in fluorescence}
        thresholds[u].update(cells=len(v), geometry_skips=dict(skipped))
        thresholds[u].update({f"{c}_mix": v[c].value_counts().to_dict() for c in mixes})
        thresholds[u].update(
            {
                f"{key}_center_p99_quantiles": dict(
                    zip(map(str, quantiles), v[f"{key}_center_p99"].quantile(quantiles))
                )
                for key in fluorescence
            }
        )
        print(u, json.dumps(thresholds[u], default=str), flush=True)
    return thresholds, pd.concat(rows, ignore_index=True)


def eligible(d, category, confidence):
    """Return the eligibility mask of one category.

    Parameters
    ----------
    d : pandas.DataFrame
        Attempt-12 joined table.
    category : str
        One of ``CATEGORIES``.
    confidence : float
        Minimum ``cycle_confidence`` for cycle-state categories.

    Returns
    -------
    pandas.Series
        Boolean mask aligned to ``d``.
    """
    if category == "UNINF":
        uninfected = d.perturbation.eq("uninfected") & d.pal_phase_score.le(UNINFECTED_PAL) & d.pal_state.eq("negative")
        return uninfected & d.sec61b_phase_score.le(NOT_REMODELED)
    infected = d.perturbation.isin(["ZIKV", "DENV"]) & d.pal_phase_score.ge(INFECTED_PAL) & d.pal_state.eq("positive")
    if category == "INFREM":
        return infected & d.sec61b_phase_score.ge(REMODELED)
    return infected & d.cycle_state.eq(category.removeprefix("INF_")) & d.cycle_confidence.ge(confidence)


def draw(
    d,
    mask,
    rng,
    used_tracks,
    stores,
    cache,
    thresholds,
    qc,
    units=DATASETS,
    unit="dataset",
    channels=CROP_CHANNELS,
    labels=None,
):
    """Round-robin random draw over units with diversity constraints and crop QC.

    Parameters
    ----------
    d : pandas.DataFrame
        Cohort table.
    mask : pandas.Series
        Category eligibility aligned to ``d``.
    rng : numpy.random.Generator
        Shared selection generator.
    used_tracks : set
        ``(dataset, fov_name, track_id)`` already used elsewhere in the gallery.
    stores, cache : dict
        Image zarr groups and position cache.
    thresholds : dict
        Per-unit QC thresholds keyed by crop channel.
    qc : tuple of str
        Crop channels whose central-112 p99 must reach the unit threshold, checked in order.
    units : list of str, optional
        Round-robin units in order (attempt 13: the datasets).
    unit : str, optional
        Column of ``d`` holding the unit.
    channels : dict, optional
        Crop channels (see ``read_crop``).
    labels : dict, optional
        Expected omero channel labels per dataset.

    Returns
    -------
    tuple of (list, collections.Counter, pandas.DataFrame)
        Accepted ``(row_index, crop)`` pairs, skip/rejection counts and the eligible pool.
    """
    pool = d[mask]
    active_units = [u for u in units if pool[unit].eq(u).any()]
    queues = {u: iter(rng.permutation(pool.index[pool[unit].eq(u)].to_numpy())) for u in active_units}
    tracks = set()
    fovs = Counter()
    stats = Counter()
    picks = []
    active = list(active_units)
    while len(picks) < PER_CATEGORY and active:
        for u in list(active):
            if len(picks) == PER_CATEGORY:
                break
            for i in queues[u]:
                r = d.loc[i]
                track = (r.dataset, r.fov_name, int(r.track_id))
                if track in used_tracks or track in tracks:
                    stats["skip_track_used"] += 1
                    continue
                if fovs[track[:2]] >= PER_FOV:
                    stats["skip_fov_full"] += 1
                    continue
                stats["crops_read"] += 1
                reason, out = read_crop(stores, cache, r, channels, labels)
                for key in qc:
                    if reason is None and out["meta"][f"{key}_center_p99"] < thresholds[u][key]:
                        reason = f"{key}_dim"
                if reason:
                    stats["reject_" + reason] += 1
                    continue
                tracks.add(track)
                fovs[track[:2]] += 1
                picks.append((i, out))
                break
            else:
                active.remove(u)
    return picks, stats, pool


def select(d, stores, cache, thresholds):
    """Select all categories, relaxing cycle confidence to 0.6 only if fewer than 10 survive.

    Parameters
    ----------
    d : pandas.DataFrame
        Attempt-12 joined table.
    stores, cache : dict
        Image zarr groups and position cache.
    thresholds : dict
        Per-dataset QC thresholds.

    Returns
    -------
    tuple of (pandas.DataFrame, dict, dict)
        Candidate table, raw crops keyed ``{ID}_{channel}`` and per-category selection report.
    """
    rng = np.random.default_rng(SEED)
    used_tracks = set()
    rows = []
    images = {}
    report = {}
    for category in CATEGORIES:
        attempts = []
        confidence = CONFIDENCE
        qc = ("tomm20",) if category == "UNINF" else ("tomm20", "pal")
        mask = eligible(d, category, confidence)
        picks, stats, pool = draw(d, mask, rng, used_tracks, stores, cache, thresholds, qc)
        attempts.append(dict(confidence=confidence, pool=len(pool), selected=len(picks), counts=dict(stats)))
        if len(picks) < PER_CATEGORY and category.startswith("INF_"):
            confidence = RELAXED_CONFIDENCE
            mask = eligible(d, category, confidence)
            picks, stats, pool = draw(d, mask, rng, used_tracks, stores, cache, thresholds, qc)
            attempts.append(dict(confidence=confidence, pool=len(pool), selected=len(picks), counts=dict(stats)))
        assert len(picks) == PER_CATEGORY, (category, attempts)
        for k, (i, out) in enumerate(picks, start=1):
            r = d.loc[i]
            cid = f"{category}-{k:02d}"
            used_tracks.add((r.dataset, r.fov_name, int(r.track_id)))
            for ch in CHANNELS:
                images[f"{cid}_{ch}"] = out["crop"][ch]
            row = dict(candidate_id=cid, category=category, selection_order=k, row_index=int(i))
            row.update(r[COLUMNS].to_dict())
            row.update(
                out["meta"],
                cycle_confidence_threshold=confidence if category.startswith("INF_") else np.nan,
                relaxed=len(attempts) > 1,
                tomm20_qc_threshold=thresholds[r.dataset]["tomm20"],
                pal_qc_threshold=thresholds[r.dataset]["pal"],
                pal_qc_applied=category != "UNINF",
                image_path=str(IMAGES / DATES[r.dataset] / f"{DATES[r.dataset]}.zarr"),
            )
            rows.append(row)
        chosen = pd.DataFrame(rows[-PER_CATEGORY:])
        report[category] = dict(
            attempts=attempts,
            confidence_used=confidence if category.startswith("INF_") else None,
            relaxed=len(attempts) > 1,
            eligible_rows=len(pool),
            eligible_by_dataset=pool.dataset.value_counts().reindex(DATASETS, fill_value=0).to_dict(),
            eligible_tracks=int(pool.groupby(["dataset", "fov_name", "track_id"]).ngroups),
            eligible_fovs=int(pool.groupby(["dataset", "fov_name"]).ngroups),
            selected_by_dataset=chosen.dataset.value_counts().reindex(DATASETS, fill_value=0).to_dict(),
            selected_by_perturbation=chosen.perturbation.value_counts().to_dict(),
            selected_fovs=int(chosen.groupby(["dataset", "fov_name"]).ngroups),
        )
        print(category, json.dumps(report[category], default=str), flush=True)
    return pd.DataFrame(rows), images, report


def contrast_limits(cands, images, channels=CHANNELS, by="dataset"):
    """Compute 1st-99.5th percentile display limits per unit and channel over all selected crops.

    Parameters
    ----------
    cands : pandas.DataFrame
        Candidate table.
    images : dict
        Raw crops keyed ``{ID}_{channel}``.
    channels : list of str, optional
        Crop channels.
    by : str, optional
        Candidate column defining the contrast unit (attempt 13: dataset).

    Returns
    -------
    dict
        ``{unit: {channel: [low, high], 'crops': n}}``.
    """
    limits = {}
    for ds, g in cands.groupby(by):
        limits[ds] = dict(crops=len(g))
        for ch in channels:
            values = np.concatenate([images[f"{cid}_{ch}"].ravel() for cid in g.candidate_id])
            limits[ds][ch] = [float(v) for v in np.percentile(values, [1, 99.5])]
    return limits


def scaled(arr, limit):
    """Scale an image to [0, 1] with fixed limits."""
    return np.clip((arr - limit[0]) / (limit[1] - limit[0]), 0, 1)


def merge(pal, tomm20, limits, organelle="tomm20"):
    """Additive pAL magenta + organelle (default TOMM20) green RGB merge, clipped to [0, 1]."""
    rgb = scaled(pal, limits["pal"])[..., None] * MAGENTA + scaled(tomm20, limits[organelle])[..., None] * GREEN
    return np.clip(rgb, 0, 1)


def rule_text(category, confidence):
    """Return the selection rule of one category as display text."""
    if category == "UNINF":
        return f"Rule: {UNINFECTED_RULE}; AND SEC61B Phase score ≤ {NOT_REMODELED}"
    if category == "INFREM":
        return f"Rule: {INFECTED_RULE}; AND SEC61B Phase score ≥ {REMODELED}"
    state = category.removeprefix("INF_")
    return f"Rule: {INFECTED_RULE}; AND PCNA→Phase cycle state = {state} with confidence ≥ {confidence}"


def render_strips(header, strips, footer, path, n_tiles=4, caption=CAPTION, footer_height=FOOTER, header_height=HEADER):
    """Render one page as 2 columns x 5 rows of tile strips with header and footer text.

    Parameters
    ----------
    header : list of tuple
        ``(y_from_top_in, text, fig.text kwargs)`` lines above the grid.
    strips : list of dict
        At most 10 strips, each with ``captions`` (``(y_offset_in, text, kwargs)`` from the
        strip top), ``tiles`` (``(label, image, imshow kwargs, scale_bar)``) and ``pixel_um``.
    footer : list of tuple
        ``(y_offset_in, text, kwargs)`` lines measured from the footer top.
    path : pathlib.Path
        Output path without suffix; ``.png`` and ``.pdf`` are written.
    n_tiles : int, optional
        Tiles per strip.
    caption : float, optional
        Caption height above each strip, inches.
    footer_height : float, optional
        Footer height, inches.
    header_height : float, optional
        Header height above the grid, inches.
    """
    strip = n_tiles * TILE + (n_tiles - 1) * GAP
    width = 2 * MARGIN + 2 * strip + COL_GAP
    height = header_height + 5 * (caption + TILE) + footer_height
    fig = plt.figure(figsize=(width, height))

    def text(x, y_from_top, s, **kw):
        fig.text(x / width, 1 - y_from_top / height, s, va="top", **kw)

    for y, s, kw in header:
        text(MARGIN, y, s, **kw)
    for k, item in enumerate(strips):
        x0 = MARGIN + (k % 2) * (strip + COL_GAP)
        top = header_height + (k // 2) * (caption + TILE)
        for y, s, kw in item["captions"]:
            text(x0, top + y, s, **kw)
        for j, (label, image, kw, scale_bar) in enumerate(item["tiles"]):
            ax = fig.add_axes(
                [(x0 + j * (TILE + GAP)) / width, 1 - (top + caption + TILE) / height, TILE / width, TILE / height]
            )
            ax.imshow(image, **kw)
            ax.set_axis_off()
            ax.text(6, 6, label, color="white", size=7, va="top", path_effects=DARK)
            if scale_bar:
                ax.plot([12, 12 + 5 / item["pixel_um"]], [210, 210], c="white", lw=2.5, solid_capstyle="butt")
                ax.text(12, 202, "5 µm", color="white", size=7, path_effects=DARK)
    for y, s, kw in footer:
        text(MARGIN, height - footer_height + y, s, **kw)
    for ext in ["png", "pdf"]:
        fig.savefig(path.parent / f"{path.name}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def render_page(category, cands, images, limits, thresholds, report, out):
    """Render one category page as 2 columns x 5 rows of 4-tile strips.

    Parameters
    ----------
    category : str
        Category name.
    cands : pandas.DataFrame
        The category's 10 candidates.
    images : dict
        Raw crops keyed ``{ID}_{channel}``.
    limits : dict
        Per-dataset display limits.
    thresholds : dict
        Per-dataset QC thresholds.
    report : dict
        Per-category selection report.
    out : pathlib.Path
        Gallery directory.
    """
    info = report[category]
    relaxed = " (RELAXED from 0.8: fewer than 10 passed at 0.8)" if info["relaxed"] else ""
    pool = ", ".join(f"{ds} {n:,}" for ds, n in info["eligible_by_dataset"].items())
    pal_qc = "; central 112×112 pAL p99 ≥ dataset threshold" if category != "UNINF" else ""
    header = [
        (0.18, f"{category}  —  {TITLES[category]}  ·  10 candidates", dict(size=14, weight="bold")),
        (0.48, rule_text(category, info["confidence_used"]) + relaxed, dict(size=9)),
        (
            0.68,
            f"Eligible rows {info['eligible_rows']:,} ({pool}). Random draw (numpy default_rng(17)) with round-robin "
            "over datasets, ≤1 cell per track across the whole gallery, ≤2 per FOV; QC-rejected crops replaced by the "
            "next draw.",
            dict(size=8),
        ),
        (
            0.86,
            "Crop QC: 224×224 crop fully in bounds; Phase focus slice z present with z ≥ 5 and z+11 ≤ Z; "
            f"central 112×112 TOMM20 p99 ≥ dataset threshold{pal_qc} (thresholds = dataset median over 300 random "
            "cells).",
            dict(size=8),
        ),
    ]
    strips = []
    for _, r in cands.iterrows():
        lim = limits[r.dataset]
        crop = {ch: images[f"{r.candidate_id}_{ch}"] for ch in CHANNELS}
        tiles = [("Phase", "phase"), ("pAL", "pal"), ("TOMM20", "tomm20")]
        strips.append(
            dict(
                captions=[
                    (
                        0.1,
                        f"{r.candidate_id}  ·  {r.dataset}  ·  {r.perturbation}  ·  {r.hours_post_perturbation:.1f} HPI"
                        f"  ·  fov {r.fov_name} / track {r.track_id} / t {r.t}",
                        dict(size=8, weight="bold"),
                    ),
                    (
                        0.27,
                        f"pAL Phase {r.pal_phase_score:.2f}  ·  pAL fluor. posterior {r.pal_posterior:.3f}  ·  "
                        f"SEC61B Phase {r.sec61b_phase_score:.2f}  ·  cycle {r.cycle_state} ({r.cycle_confidence:.2f})",
                        dict(size=8),
                    ),
                ],
                tiles=[
                    (label, crop[ch], dict(cmap=CMAPS[ch], vmin=lim[ch][0], vmax=lim[ch][1]), label == "Phase")
                    for label, ch in tiles
                ]
                + [("merge", merge(crop["pal"], crop["tomm20"], lim), {}, True)],
                pixel_um=r.pixel_um,
            )
        )
    footer = [
        (
            0.1,
            "Display contrast fixed per dataset and channel across the whole 60-crop gallery (1st–99.5th percentile "
            "over all selected crops of that dataset). Phase: focus slice; pAL, TOMM20: 16-slice MIP (z−5…z+10); "
            "merge uses the same pAL/TOMM20 limits.",
            dict(size=8),
        )
    ]
    for n, ds in enumerate(DATASETS):
        lim = limits[ds]
        footer.append(
            (
                0.29 + n * 0.17,
                f"{ds} ({lim['crops']} gallery crops): Phase [{lim['phase'][0]:.4g}, {lim['phase'][1]:.4g}]  ·  "
                f"pAL [{lim['pal'][0]:.4g}, {lim['pal'][1]:.4g}]  ·  "
                f"TOMM20 [{lim['tomm20'][0]:.4g}, {lim['tomm20'][1]:.4g}]"
                f"   |   QC thresholds: TOMM20 p99 ≥ {thresholds[ds]['tomm20']:.4g}, pAL p99 ≥ {thresholds[ds]['pal']:.4g}",
                dict(size=8),
            )
        )
    render_strips(header, strips, footer, out / category)


def render_overview(
    display,
    cands,
    out,
    xy=("Phase3D_umap1", "Phase3D_umap2"),
    background="attempt-12 display",
    axis="Phase3D UMAP",
    title="Figure 6 crop candidates on the shared Phase UMAP (60 candidates, 10 per category)",
    categories=CATEGORIES,
    titles=TITLES,
    shapes=None,
    annotate=True,
    stem="overview_umap",
):
    """Render a UMAP display sample in gray with candidates as filled, white-edged markers.

    Parameters
    ----------
    display : pandas.DataFrame
        Background display sample (attempt 13: attempt-12 display rows).
    cands : pandas.DataFrame
        Candidate table.
    out : pathlib.Path
        Gallery directory.
    xy : tuple of str, optional
        UMAP coordinate columns shared by ``display`` and ``cands``.
    background : str, optional
        Legend label of the gray background.
    axis : str, optional
        Axis-label prefix.
    title : str, optional
        Axes title.
    categories : list of str, optional
        Categories in draw and legend order; colours from ``CATEGORY_COLORS``.
    titles : dict, optional
        Category -> legend description.
    shapes : tuple of (str, dict), optional
        Candidate column and value -> marker; adds one legend entry per value.
    annotate : bool, optional
        Label each candidate with its ID.
    stem : str, optional
        Output file stem.
    """
    x, y = xy
    fig, ax = plt.subplots(figsize=(9, 8.5), layout="constrained")
    ax.scatter(display[x], display[y], c="#cccccc", s=1, linewidths=0, rasterized=True)
    handles = [
        Line2D([], [], marker="o", ls="", color="#cccccc", markersize=4, label=f"{background} ({len(display):,})")
    ]
    for category in categories:
        g = cands[cands.category.eq(category)]
        color = CATEGORY_COLORS[category]
        groups = [("o", g)] if shapes is None else [(shapes[1][v], h) for v, h in g.groupby(shapes[0])]
        for marker, h in groups:
            ax.scatter(h[x], h[y], s=40, c=color, marker=marker, edgecolors="white", linewidths=0.7, zorder=3)
        if annotate:
            for _, r in g.iterrows():
                ax.annotate(
                    r.candidate_id,
                    (r[x], r[y]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    path_effects=HALO,
                    zorder=4,
                )
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=7,
                label=f"{category}: {titles[category]}",
            )
        )
    if shapes is not None:
        handles += [
            Line2D([], [], marker=m, ls="", color="#555555", markersize=7, label=v) for v, m in shapes[1].items()
        ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=7, frameon=False)
    ax.set(xticks=[], yticks=[], xlabel=f"{axis} 1", ylabel=f"{axis} 2", title=title)
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"{stem}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)


def write_readme(cands, thresholds, report, out):
    """Write the gallery README with rules, thresholds and candidate IDs.

    Parameters
    ----------
    cands : pandas.DataFrame
        Candidate table.
    thresholds : dict
        Per-dataset QC thresholds.
    report : dict
        Per-category selection report.
    out : pathlib.Path
        Gallery directory.
    """
    lines = [
        "# Figure 6 crop-candidate gallery (attempt 13)",
        "",
        f"Review gallery of 60 candidate crops drawn from `{TABLE}`. "
        "No model fitting; scores and labels are the attempt-12 values.",
        "",
        "## Category rules",
        "",
        f"- {INFECTED_RULE}.",
        f"- {UNINFECTED_RULE}.",
        "- `INFREM`: infected AND SEC61B Phase score ≥ 0.8.",
        "- `UNINF`: uninfected AND SEC61B Phase score ≤ 0.2.",
        "- `INF_G1`, `INF_S`, `INF_G2`, `INF_M`: infected AND PCNA→Phase `cycle_state` = state AND "
        "`cycle_confidence` ≥ 0.8 (relaxed to ≥ 0.6 only if fewer than 10 pass QC).",
        "",
        "Infected cells must have Phase-predicted and fluorescence-measured pAL agree, and uninfected cells must agree "
        "on negative. These candidates are therefore selected agreeing exemplars, not random cells, and their rate "
        "does not represent head accuracy.",
        "",
        "## Selection and crop QC",
        "",
        "- Random draw with `numpy.random.default_rng(17)` from each category's eligible pool (scores are not ranked), "
        "round-robin over the datasets with eligible cells, at most one cell per (dataset, fov_name, track_id) across "
        "the whole gallery, at most 2 per FOV per category. A QC rejection draws a replacement.",
        "- QC: 224×224 crop fully in bounds; Phase focus slice z present with z ≥ 5 and z+11 ≤ Z; central 112×112 "
        "TOMM20 16-slice-MIP p99 ≥ the dataset threshold; for infected categories also central 112×112 pAL MIP p99 ≥ "
        "the dataset threshold.",
        "- Thresholds are each dataset's median central-112 p99 over 300 random geometrically valid cells (a separate "
        "`default_rng(17)` draw over all of the dataset's rows; `source_data/qc_reference_sample.csv`).",
        "",
        "| dataset | TOMM20 p99 threshold | pAL p99 threshold |",
        "|---|---|---|",
    ]
    lines += [f"| {ds} | {thresholds[ds]['tomm20']:.4g} | {thresholds[ds]['pal']:.4g} |" for ds in DATASETS]
    lines += ["", "## Candidates (pick by ID)", ""]
    for category in CATEGORIES:
        info = report[category]
        relaxed = f", cycle confidence relaxed to ≥ {info['confidence_used']}" if info["relaxed"] else ""
        mix = ", ".join(f"{ds} {n}" for ds, n in info["selected_by_dataset"].items())
        lines.append(f"- `{category}` ({TITLES[category]}; {mix}{relaxed}):")
        for _, r in cands[cands.category.eq(category)].iterrows():
            lines.append(
                f"  - `{r.candidate_id}`: {r.dataset}, {r.perturbation}, {r.fov_name} track {r.track_id} t {r.t}"
            )
    lines += [
        "",
        "Reply with chosen IDs (e.g. `INFREM-03, UNINF-07, INF_G1-02, INF_S-05, INF_G2-01, INF_M-04`). Crops and keys "
        "for each ID are in `candidates.csv` and `source_data/candidate_crops.npz` (keys `{ID}_phase`, `{ID}_pal`, "
        "`{ID}_tomm20`; raw, unscaled).",
        "",
        "## Files",
        "",
        "- `INFREM.png/.pdf` … `INF_M.png/.pdf`: one page per category, Phase | pAL | TOMM20 | merge.",
        "- `overview_umap.png/.pdf`: Phase UMAP, attempt-12 display sample in gray, candidates as filled markers.",
        "- `candidates.csv`, `source_data/`, `manifest.json`, `code/`.",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))


def main():
    """Claim the gallery directory, select candidates, render pages and write records."""
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "source_data").mkdir()
    (OUTPUT / "code").mkdir()
    d = pd.read_parquet(TABLE)
    d["fov_name"] = d.fov_name.astype(str)
    assert len(d) == 184514 and not d.duplicated(["dataset", "fov_name", "track_id", "t", "id"]).any()
    display = pd.read_parquet(DISPLAY)
    channels = confirm_channels(d)
    print("Channels confirmed", json.dumps(channels), flush=True)
    stores = {ds: zarr.open_group(str(IMAGES / DATES[ds] / f"{DATES[ds]}.zarr"), mode="r") for ds in DATASETS}
    cache = {}
    thresholds, reference = reference_thresholds(d, stores, cache)
    reference.to_csv(OUTPUT / "source_data/qc_reference_sample.csv", index=False)
    cands, images, report = select(d, stores, cache, thresholds)
    assert cands[["Phase3D_umap1", "Phase3D_umap2"]].notna().all().all()
    assert not cands.duplicated(["dataset", "fov_name", "track_id"]).any()
    limits = contrast_limits(cands, images)
    print("Contrast limits", json.dumps(limits), flush=True)
    cands.to_csv(OUTPUT / "candidates.csv", index=False)
    np.savez_compressed(OUTPUT / "source_data/candidate_crops.npz", **images)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    for category in CATEGORIES:
        render_page(category, cands[cands.category.eq(category)], images, limits, thresholds, report, OUTPUT)
    render_overview(display, cands, OUTPUT)
    write_readme(cands, thresholds, report, OUTPUT)
    scripts = [Path(__file__), Path(prototype.__file__)]
    for p in scripts:
        shutil.copyfile(p, OUTPUT / "code" / p.name)
    write_json(
        OUTPUT / "manifest.json",
        dict(
            sources={str(TABLE): sha(TABLE), str(DISPLAY): sha(DISPLAY)},
            script_sha256={p.name: sha(p) for p in scripts},
            source_attempt=str(ATTEMPT12),
            image_zarrs={ds: str(IMAGES / DATES[ds] / f"{DATES[ds]}.zarr") for ds in DATASETS},
            channel_confirmation=dict(zarr_channels=[TOMM20, PAL, PHASE], pal=PAL, tomm20=TOMM20, per_dataset=channels),
            seed=SEED,
            rng=(
                "selection: one numpy default_rng(17) shared across categories in order "
                f"{CATEGORIES}, permuting each dataset's eligible rows; QC reference: separate default_rng(17)"
            ),
            rules=dict(
                infected=INFECTED_RULE,
                uninfected=UNINFECTED_RULE,
                INFREM="infected AND sec61b_phase_score >= 0.8",
                UNINF="uninfected AND sec61b_phase_score <= 0.2",
                cycle="infected AND cycle_state == X AND cycle_confidence >= 0.8 (0.6 only if < 10 pass)",
                diversity=(
                    "round-robin over datasets with eligible rows; <= 1 per (dataset, fov_name, track_id) "
                    "gallery-wide; <= 2 per (dataset, fov_name) per category"
                ),
                per_category=PER_CATEGORY,
            ),
            crop_qc=dict(
                crop_px=2 * HALF,
                bounds="y-112 >= 0, x-112 >= 0, y+112 <= Y, x+112 <= X",
                focus="focus_slice.Phase3D.per_timepoint[t] present, focus >= 5, focus + 11 <= Z",
                tomm20="central 112x112 p99 of GFP 16-slice MIP >= dataset threshold",
                pal="central 112x112 p99 of mCherry 16-slice MIP >= dataset threshold (infected categories only)",
                threshold_rule=f"dataset median central-112 p99 over {REFERENCE_CELLS} random geometrically valid cells",
                thresholds=thresholds,
            ),
            selection=report,
            contrast_limits=dict(rule="1st-99.5th percentile over all selected crops of each dataset", limits=limits),
            outputs=sorted(str(p.relative_to(OUTPUT)) for p in OUTPUT.rglob("*") if p.is_file()),
        ),
    )
    print(cands[["candidate_id", "dataset", "perturbation", "fov_name", "track_id", "t"]].to_string(index=False))
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
