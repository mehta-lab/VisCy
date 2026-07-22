"""Occlusion-attribution report for a DynaCLR classifier over random cell tracks.

Given a dataset (raw OME-Zarr with ``focus_slice`` + ``normalization`` zattrs),
a tracking OME-Zarr (label images), a DynaCLR encoder (config + checkpoint),
and a linear classifier joblib (optionally with a separate StandardScaler),
this picks a few random cell tracks, scores the classifier probability along
each track over time, runs occlusion saliency per frame, and writes a report
(PNG + PDF): per-cell clean-phase row + occlusion-overlay row, plus a shared
P(target) vs time plot.

Preprocessing matches DynaCLR training (validated against stored predict-zarr
embeddings): per-timepoint z-score normalization, per-timepoint focus z-slice
(no MIP for phase), and pixel-size rescale (crop the physical-area-matched
window then resize to the model's patch size).

Not a CLI yet — edit ``CONFIG`` below and run:
    uv run python applications/dynaclr/scripts/occlusion_infection_report.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from iohub.ngff import open_ome_zarr
from scipy import ndimage as ndi

from dynaclr.visualization.compare_pca_rgb_cli import _build_dynaclr
from dynaclr.visualization.occlusion_overlay import make_embed_fn
from viscy_utils.visualization.occlusion import occlusion_saliency, saliency_to_rgb

# ── CONFIG — edit these ────────────────────────────────────────────────────────
CONFIG: dict[str, Any] = {
    # data
    "data_zarr": "/hpc/projects/organelle_phenotyping/datasets/2026_04_10_A549_TOMM20_ZIKV/2026_04_10_A549_TOMM20_ZIKV.zarr",
    "tracking_zarr": "/hpc/projects/organelle_phenotyping/datasets/2026_04_10_A549_TOMM20_ZIKV/tracking.zarr",
    "wells": None,  # None = all wells; or a list like ["B/2", "B/4"] to restrict
    # model
    "config_path": "/hpc/projects/organelle_phenotyping/models/DynaCLR-2D-MIP-BagOfChannels/2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/config.yaml",
    "ckpt_path": "/hpc/projects/organelle_phenotyping/models/DynaCLR-2D-MIP-BagOfChannels/2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/DynaCLR-2D-MIP-BagOfChannels/jbrwhzr3/checkpoints/epoch=105-step=84800.ckpt",
    # classifier (joblib is a sklearn estimator or a LinearClassifierPipeline)
    "classifier_path": "/hpc/projects/organelle_phenotyping/models/linear_classifiers/DynaCLR-2D-MIP-BagOfChannels-single-marker-fix-shuffler/infectomics_epoch105_step84800/v3/remodeling_state_Phase3D_ZIKV_Mantisv2.joblib",
    "scaler_path": "/hpc/mydata/soorya.pradeep/GitHub/dynaclr-organelle-paper/organelle_remodeling_paper/figures/label-free_classifier_infection/sec61b_timelapse_zikv/phase_scaler.joblib",  # None if the joblib needs no external scaler
    "target_class": 1,  # class label whose probability is the saliency target + plotted P
    # preprocessing — read from config.yaml by default; set here to override.
    "image_channel": None,  # None = derive from config (focus_channel / normalizations)
    "final_patch": None,  # None = data.final_yx_patch_size from config
    "reference_pixel_size_xy_um": None,  # None = data.reference_pixel_size_xy_um from config
    "norm_level": None,  # None = level from config normalizations (e.g. timepoint_statistics)
    # occlusion
    "occ_size": 8,
    "stride": 4,
    "fill": "zero",  # occluder fill: "zero" | "mean" | "blur" | float. "blur" preserves
    # gross shape/density and removes only fine texture (faithful for phase); "zero"/"mean"
    # erase to background (≈0 under z-score). See occlusion_saliency docstring.
    "distance": "signed_delta",  # signed_delta (classifier) | l2 | cosine (embedding)
    "cmap": "icefire",
    "clip_value": 0.2,
    # sampling
    "n_cells": 4,  # 3–5 random tracks
    "min_track_len": 20,
    "seed": 0,
    # output
    "out_dir": "applications/dynaclr/scripts/explainability/occlusion/output",
    "out_stem": "occlusion_infection_report",
    "n_show": 6,  # montage columns per cell
}
# ───────────────────────────────────────────────────────────────────────────────


def _read_model_config(config_path: str) -> dict[str, Any]:
    """Pull preprocessing metadata from the training config.yaml ``data:`` block.

    Returns final_patch, reference_pixel_size_xy_um, norm_level, and image_channel.
    These are the source of truth — they are NOT stored in the checkpoint.
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]["init_args"]
    final_yx = data.get("final_yx_patch_size", [160, 160])
    norms = data.get("normalizations") or []
    norm_level = None
    for n in norms:
        ia = n.get("init_args", {})
        if "level" in ia:
            norm_level = ia["level"]
            break
    # image channel: prefer explicit focus_channel, else first normalization key
    channel = data.get("focus_channel")
    if channel is None and norms:
        keys = norms[0].get("init_args", {}).get("keys") or []
        channel = keys[0] if keys else None
    return {
        "final_patch": int(final_yx[-1]),
        "reference_pixel_size_xy_um": data.get("reference_pixel_size_xy_um"),
        "norm_level": norm_level or "timepoint_statistics",
        "image_channel": channel,
    }


def _classifier_closure(clf_path: str, scaler_path: str | None, target_class, embed_fn):
    """Return (fn: images -> (B,1) P(target_class), classes)."""
    obj = joblib.load(clf_path)
    # LinearClassifierPipeline has .classifier / .predict_proba; a bare sklearn
    # estimator exposes predict_proba / classes_ directly.
    estimator = getattr(obj, "classifier", obj)
    classes = list(estimator.classes_)
    tgt = classes.index(target_class)
    scaler = joblib.load(scaler_path) if scaler_path else None

    def fn(x: torch.Tensor) -> torch.Tensor:
        emb = embed_fn(x).detach().cpu().float().numpy()
        if scaler is not None:
            emb = scaler.transform(emb)
        proba = (
            obj.predict_proba(emb) if scaler is None and hasattr(obj, "predict_proba") else estimator.predict_proba(emb)
        )
        return torch.from_numpy(np.asarray(proba)[:, tgt : tgt + 1]).to(x.device, torch.float32)

    return fn, classes


def _pick_random_tracks(tracking_zarr: str, data_zarr: str, wells, n: int, min_len: int, half: int, seed: int):
    """Return list of (well, fov, label_id). In-bounds, long-enough, random, ideally spread across wells."""
    rng = np.random.default_rng(seed)
    g = zarr.open(tracking_zarr, mode="r")
    with open_ome_zarr(data_zarr, mode="r") as plate:
        data_fovs = {name for name, _ in plate.positions()}
    candidates = []
    # tracking OME-Zarr is nested row/col/fov (e.g. "A/2/0000").
    for row in g.keys():
        for col in g[row].keys():
            well = f"{row}/{col}"
            for fov_name in g[row][col].keys():
                fov = f"{well}/{fov_name}"
                if fov not in data_fovs:
                    continue
                if wells is not None and well not in wells and fov not in wells:
                    continue
                arr = g[f"{fov}/0"]
                T, H, W = arr.shape[0], arr.shape[3], arr.shape[4]
                lbl0 = np.asarray(arr[0, 0, 0])
                lblL = np.asarray(arr[T - 1, 0, 0])
                common = np.intersect1d(np.unique(lbl0), np.unique(lblL))
                common = common[common > 0]
                for lid in common:
                    present, ok = 0, True
                    for t in range(0, T, 6):
                        m = np.asarray(arr[t, 0, 0]) == lid
                        if m.any():
                            present += 1
                            cy, cx = ndi.center_of_mass(m)
                            if not (half <= cy < H - half and half <= cx < W - half):
                                ok = False
                                break
                    if ok and present >= max(min_len // 6, 3):
                        candidates.append((well, fov_name, int(lid)))
    if not candidates:
        raise RuntimeError("no in-bounds tracks found; loosen min_track_len or wells filter")
    rng.shuffle(candidates)
    # spread across distinct wells first
    picked, seen_wells = [], set()
    for c in candidates:
        if c[0] not in seen_wells:
            picked.append(c)
            seen_wells.add(c[0])
        if len(picked) >= n:
            break
    for c in candidates:
        if len(picked) >= n:
            break
        if c not in picked:
            picked.append(c)
    return picked[:n]


def _load_track(data_zarr, tracking_zarr, well, fov, label, image_channel, final, half, norm_level):
    """Return (imgs (Tc,1,final,final) z-scored+resized, times).

    Note: this takes the per-timepoint FOCUS Z-SLICE, a validated approximation
    (cosine >0.997 vs stored predict-zarr embeddings) of the datamodule's full
    z_extraction_window + z_reduction. Exact for 2D / in_stack_depth=1 models
    where the phase channel is a single focus/center slice.
    """
    lbl_arr = zarr.open(tracking_zarr, mode="r")[f"{well}/{fov}/0"]
    T = lbl_arr.shape[0]
    crops, times = [], []
    with open_ome_zarr(f"{data_zarr}/{well}/{fov}", mode="r") as pos:
        cidx = pos.channel_names.index(image_channel)
        arr = pos["0"]
        H, W = arr.shape[3], arr.shape[4]
        nlevel = pos.zattrs["normalization"][image_channel][norm_level]
        nmeta = nlevel if norm_level == "timepoint_statistics" else {"_": nlevel}
        focus = pos.zattrs["focus_slice"][image_channel]["per_timepoint"]
        for t in range(T):
            m = np.asarray(lbl_arr[t, 0, 0]) == label
            if not m.any():
                continue
            cy, cx = (int(round(v)) for v in ndi.center_of_mass(m))
            cy = min(max(cy, half), H - half)
            cx = min(max(cx, half), W - half)
            z = int(focus[str(t)])
            patch = arr[t, cidx, z, cy - half : cy + half, cx - half : cx + half]
            if float(np.std(patch)) < 1e-6 or not np.isfinite(patch).all():
                continue  # skip empty/corrupt frames
            st = nmeta[str(t)] if norm_level == "timepoint_statistics" else nmeta["_"]
            mean, std = float(st["mean"]), max(float(st["std"]), 1e-8)
            crops.append((torch.from_numpy(patch.copy()).float()[None] - mean) / std)
            times.append(t)
    imgs = torch.stack(crops, dim=0)
    imgs = F.interpolate(imgs, size=(final, final), mode="bilinear", align_corners=False)
    return imgs, times


def main(cfg: dict[str, Any]) -> None:
    """Run the occlusion report: pick tracks, score + occlude, write PNG/PDF."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocessing metadata: read from the training config.yaml, CONFIG overrides.
    meta = _read_model_config(cfg["config_path"])
    final = int(cfg["final_patch"] if cfg.get("final_patch") is not None else meta["final_patch"])
    ref_px = (
        cfg["reference_pixel_size_xy_um"]
        if cfg.get("reference_pixel_size_xy_um") is not None
        else meta["reference_pixel_size_xy_um"]
    )
    norm_level = cfg["norm_level"] or meta["norm_level"]
    image_channel = cfg["image_channel"] or meta["image_channel"]
    print(f"from config: final_patch={final}, ref_px={ref_px}, norm_level={norm_level}, image_channel={image_channel}")

    # pixel-size rescale: crop physical-area-matched window then resize to `final`.
    with open_ome_zarr(cfg["data_zarr"], mode="r") as _plate:
        inf_px = float(next(iter(_plate.positions()))[1].scale[-1])
    if ref_px:
        scale = float(ref_px) / inf_px
        initial = max(2, 2 * round(final * scale / 2))
    else:
        initial = final
    half = initial // 2
    print(f"pixel rescale: ref={ref_px} inf={inf_px:.4f} -> crop {initial}px resize to {final}px")

    model = _build_dynaclr({"config_path": cfg["config_path"], "ckpt_path": cfg["ckpt_path"]}).to(device).eval()
    embed_fn = make_embed_fn(model)
    clf_fn, classes = _classifier_closure(cfg["classifier_path"], cfg.get("scaler_path"), cfg["target_class"], embed_fn)
    print(
        f"classifier classes={classes}, target={cfg['target_class']}, scaler={'yes' if cfg.get('scaler_path') else 'no'}"
    )

    picked = _pick_random_tracks(
        cfg["tracking_zarr"],
        cfg["data_zarr"],
        cfg.get("wells"),
        int(cfg["n_cells"]),
        int(cfg["min_track_len"]),
        half,
        int(cfg["seed"]),
    )
    print("picked tracks:", picked)

    tracks = []
    for well, fov, label in picked:
        imgs, times = _load_track(
            cfg["data_zarr"], cfg["tracking_zarr"], well, fov, label, image_channel, final, half, norm_level
        )
        imgs = imgs.to(device)
        proba = clf_fn(imgs).flatten().cpu().numpy()
        sal = occlusion_saliency(
            imgs,
            clf_fn,
            occ_size=int(cfg["occ_size"]),
            stride=int(cfg["stride"]),
            fill_value=cfg.get("fill", "zero"),
            batch_size=64,
            distance=cfg["distance"],
        )
        rgb = saliency_to_rgb(
            sal, cmap=cfg["cmap"], clip_value=cfg["clip_value"], upsample_to=(final, final), interp_mode="bilinear"
        )
        tracks.append({"label": f"{well}/{fov} L{label}", "imgs": imgs, "times": times, "proba": proba, "rgb": rgb})
        print(f"  {well}/{fov} L{label}: {len(times)} frames, P {proba.min():.3f}..{proba.max():.3f}")

    _plot(tracks, cfg)


def _plot(tracks, cfg):
    import matplotlib.pyplot as plt

    n_show = int(cfg["n_show"])
    n_cells = len(tracks)
    img_rows = 2 * n_cells
    fig = plt.figure(figsize=(2.0 * n_show, 2.4 * img_rows + 3), dpi=150)
    gs = fig.add_gridspec(img_rows + 1, n_show, height_ratios=[*([1.0] * img_rows), 1.6])
    for ci, tr in enumerate(tracks):
        times = tr["times"]
        idxs = np.linspace(0, len(times) - 1, n_show).round().astype(int)
        for c, i in enumerate(idxs):
            img = tr["imgs"][i, 0].cpu().numpy()
            axc = fig.add_subplot(gs[2 * ci, c])
            axc.imshow(img, cmap="gray")
            axc.set_xticks([])
            axc.set_yticks([])
            axc.set_title(f"t={times[i]} P={tr['proba'][i]:.2f}", fontsize=7)
            if c == 0:
                axc.set_ylabel(f"{tr['label']}\nphase", fontsize=7)
            axo = fig.add_subplot(gs[2 * ci + 1, c])
            axo.imshow(img, cmap="gray")
            axo.imshow(tr["rgb"][i].cpu().numpy().transpose(1, 2, 0), alpha=0.55, interpolation="nearest")
            axo.set_xticks([])
            axo.set_yticks([])
            if c == 0:
                axo.set_ylabel("+occlusion", fontsize=7)
    axp = fig.add_subplot(gs[img_rows, :])
    for tr in tracks:
        axp.plot(tr["times"], tr["proba"], "-o", ms=3, label=tr["label"])
    axp.set_ylim(-0.02, 1.02)
    axp.axhline(0.5, color="grey", lw=0.7, ls="--")
    axp.set_xlabel("timepoint")
    axp.set_ylabel(f"P(class={cfg['target_class']})")
    axp.legend(fontsize=7)
    fig.suptitle(
        f"Occlusion over time — {Path(cfg['data_zarr']).stem} | occ{cfg['occ_size']}/stride{cfg['stride']}",
        fontsize=10,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"{cfg['out_stem']}.{ext}"
        fig.savefig(path)
        print("wrote", path)


if __name__ == "__main__":
    main(CONFIG)
