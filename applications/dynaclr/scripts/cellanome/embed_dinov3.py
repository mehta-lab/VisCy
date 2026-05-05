"""Extract DINOv3 embeddings for cellanome cells → cell-level AnnData.

Reads primary_analysis.csv from the Cellanome pipeline, crops cell patches
from the OME-Zarr store, runs them through a frozen DINOv3 model, and writes
a new cell-level AnnData zarr where each row is one segmented cell.

Usage
-----
uv run python embed_dinov3.py config.yaml
"""

import argparse
import logging
import math
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import zarr
from tqdm import tqdm

from viscy_models.foundation import DINOv3Model

CHANNEL_SHORT_NAMES = {
    "White": "BF",
    "Blue-FITC (520)": "FITC",
    "Red-CY5 (700)": "CY5",
    "Green-CY3 (605)": "CY3",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_primary_analysis(
    analysis_base: str,
    scan_ids: list[int] | None = None,
    lane_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Load and concatenate primary_analysis.csv for all scans/lanes.

    Parameters
    ----------
    analysis_base : str
        Path to the image_analysis_output directory.
    scan_ids : list[int] or None
        Scan IDs to include. If None, auto-discover.
    lane_ids : list[int] or None
        Lane IDs to include. If None, auto-discover.

    Returns
    -------
    pd.DataFrame
        Concatenated primary analysis with all columns.
    """
    base = Path(analysis_base)
    if scan_ids is None:
        scan_ids = sorted(int(p.name.split("_")[1]) for p in base.glob("scan_*") if p.is_dir())
    if lane_ids is None:
        all_lanes = set()
        for scan_id in scan_ids:
            scan_dir = base / f"scan_{scan_id}"
            all_lanes.update(int(p.name.split("_")[1]) for p in scan_dir.glob("lane_*") if p.is_dir())
        lane_ids = sorted(all_lanes)

    frames = []
    for scan_id in scan_ids:
        for lane_id in lane_ids:
            csv_path = (
                base
                / f"scan_{scan_id}"
                / f"lane_{lane_id}"
                / "processed"
                / "CAGE_REGISTRATION"
                / "primary_analysis.csv"
            )
            if not csv_path.exists():
                logger.warning(f"Missing: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            frames.append(df)
            logger.info(f"scan_{scan_id}/lane_{lane_id}: {len(df)} objects")

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total: {len(combined)} objects across {len(frames)} scan/lane combinations")
    return combined


def derive_zarr_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Derive zarr position and path from cage_crop_file_name.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: cage_crop_file_name, lane_id, scan_id.

    Returns
    -------
    pd.DataFrame
        With added zarr_position and zarr_path columns.
    """

    def _parse_position(cage_crop: str) -> str:
        parts = str(cage_crop).split("_")
        return f"{parts[4]}{parts[5]}"

    df["zarr_position"] = df["cage_crop_file_name"].apply(_parse_position)
    df["zarr_path"] = df["lane_id"].astype(str) + "/" + df["scan_id"].astype(str) + "/" + df["zarr_position"]
    return df


def build_barcode_lookup(anndata_path: str) -> dict[tuple[int, str], list[str]]:
    """Build (global_cage_id_matched, lane) → [barcode_index, ...] lookup.

    Parameters
    ----------
    anndata_path : str
        Path to the transcriptome AnnData zarr.

    Returns
    -------
    dict[tuple[int, str], list[str]]
        Mapping from (cage_id, lane_string) to list of barcode obs_names.
    """
    adata = ad.read_zarr(anndata_path)
    obs = adata.obs.copy()
    obs["_lane"] = obs.index.str.extract(r"(lane_\d)")[0].to_numpy()
    obs["_cage_id"] = obs["global.cage.id.matched"].astype(int)

    lookup: dict[tuple[int, str], list[str]] = {}
    for idx, row in obs.iterrows():
        key = (row["_cage_id"], row["_lane"])
        lookup.setdefault(key, []).append(idx)

    logger.info(f"Barcode lookup: {len(lookup)} unique (cage, lane) pairs from {adata.n_obs} barcodes")
    return lookup


def join_barcodes(df: pd.DataFrame, lookup: dict[tuple[int, str], list[str]]) -> pd.DataFrame:
    """Join barcode indices to cells via (global_cage_id_matched, lane).

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: global_cage_id_matched, lane_id.
    lookup : dict
        From build_barcode_lookup.

    Returns
    -------
    pd.DataFrame
        With added barcode_index and in_anndata columns.
    """
    barcode_indices = []
    for _, row in df.iterrows():
        key = (int(row["global_cage_id_matched"]), f"lane_{int(row['lane_id'])}")
        barcodes = lookup.get(key, [])
        barcode_indices.append(";".join(barcodes) if barcodes else "")
    df["barcode_index"] = barcode_indices
    df["in_anndata"] = df["barcode_index"] != ""
    return df


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply column-level filters to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    filters : dict
        Mapping of column_name → {min, max, eq, isin}.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    for col, conditions in filters.items():
        if col not in df.columns:
            raise ValueError(f"Filter column '{col}' not found. Available: {list(df.columns)[:10]}...")
        if "min" in conditions:
            df = df[df[col] >= conditions["min"]]
        if "max" in conditions:
            df = df[df[col] <= conditions["max"]]
        if "eq" in conditions:
            df = df[df[col] == conditions["eq"]]
        if "isin" in conditions:
            df = df[df[col].isin(conditions["isin"])]
    return df


def resolve_channel_indices(store: zarr.Group, zarr_path: str, channel_names: list[str]) -> list[int]:
    """Resolve integer indices for named channels in an OME-Zarr FOV.

    Parameters
    ----------
    store : zarr.Group
        Opened zarr store.
    zarr_path : str
        Relative path to the FOV group.
    channel_names : list[str]
        Channel labels to look up.

    Returns
    -------
    list[int]
        Zero-based channel indices.
    """
    fov_group = store[zarr_path]
    channels = fov_group.attrs["omero"]["channels"]
    labels = [ch.get("label", ch.get("name", "")) for ch in channels]
    indices = []
    for name in channel_names:
        if name not in labels:
            raise ValueError(f"Channel '{name}' not found. Available: {labels}")
        indices.append(labels.index(name))
    return indices


def crop_cell(
    fov_array: np.ndarray,
    cy: int,
    cx: int,
    half: int,
    channels: list[int] | None = None,
) -> np.ndarray | None:
    """Crop a square patch centered on (cy, cx) from a 2D FOV array.

    Parameters
    ----------
    fov_array : np.ndarray
        FOV image array of shape ``(C, H, W)``.
    cy : int
        Y centroid in FOV pixels.
    cx : int
        X centroid in FOV pixels.
    half : int
        Half the crop size in pixels.
    channels : list[int] or None
        Channel indices to select. If None, use all channels.

    Returns
    -------
    np.ndarray or None
        Cropped patch, or None if out of bounds.
    """
    _, h, w = fov_array.shape
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return None
    patch = fov_array[:, y0:y1, x0:x1]
    if channels is not None:
        patch = patch[channels]
    return patch


def main():
    """Extract DINOv3 embeddings for cellanome cells."""
    parser = argparse.ArgumentParser(description="Extract DINOv3 embeddings for cellanome cells.")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    zarr_store = cfg["zarr_store"]
    analysis_base = cfg["analysis_base"]
    transcriptome_anndata = cfg.get("transcriptome_anndata", None)
    output_path = cfg["output_path"]
    model_name = cfg.get("model_name", "facebook/dinov2-base")
    channels = cfg.get("channels", None)
    output_key = cfg.get("output_key", None)
    patch_size = cfg.get("patch_size", 96)
    reference_pixel_size = cfg.get("reference_pixel_size", 1.0)
    source_pixel_size = cfg.get("source_pixel_size", 1.0)
    batch_size = cfg.get("batch_size", 128)
    device_str = cfg.get("device", "cuda")
    scan_ids = cfg.get("scan_ids", None)
    lane_ids = cfg.get("lane_ids", None)
    filters = cfg.get("filters", {})

    # --- Load and prepare data ---
    df = load_primary_analysis(analysis_base, scan_ids, lane_ids)
    n_raw = len(df)
    df = apply_filters(df, filters)
    logger.info(f"After filtering: {len(df)} cells (removed {n_raw - len(df)})")

    df = derive_zarr_paths(df)
    if transcriptome_anndata is not None:
        lookup = build_barcode_lookup(transcriptome_anndata)
        df = join_barcodes(df, lookup)
        n_matched = df["in_anndata"].sum()
        logger.info(f"Barcode match: {n_matched}/{len(df)} cells ({100 * n_matched / len(df):.1f}%)")
    else:
        logger.info("No transcriptome_anndata provided; skipping barcode join")

    # --- Pixel size rescaling ---
    # raw_crop covers the same physical area as patch_size at reference resolution.
    # Larger source pixels → fewer pixels needed.
    raw_half = round(patch_size * reference_pixel_size / source_pixel_size) // 2
    raw_crop_size = 2 * raw_half
    logger.info(f"Raw crop: {raw_crop_size}x{raw_crop_size} -> model input: {patch_size}x{patch_size}")

    # --- Resolve channels ---
    store = zarr.open(zarr_store, mode="r")
    first_zarr_path = df["zarr_path"].iloc[0]
    if channels is not None:
        channel_indices = resolve_channel_indices(store, first_zarr_path, channels)
        channel_labels = channels
    else:
        fov_group = store[first_zarr_path]
        omero_channels = fov_group.attrs["omero"]["channels"]
        channel_labels = [ch.get("label", ch.get("name", "")) for ch in omero_channels]
        channel_indices = list(range(len(channel_labels)))
    logger.info(f"Channels: {channel_labels} (indices {channel_indices})")

    short_names = [CHANNEL_SHORT_NAMES.get(ch, ch) for ch in channel_labels]
    output_key = output_key or "dinov3_" + "_".join(short_names)

    # --- Load model ---
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = DINOv3Model(model_name=model_name, freeze=True)
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded DINOv3 {model_name} on {device}")

    # --- Inference ---
    df = df.sort_values("zarr_path").reset_index(drop=True)
    current_fov_path: str | None = None
    current_fov: np.ndarray | None = None
    all_embeddings = []
    valid_indices = []
    skipped_border = 0

    n_batches = math.ceil(len(df) / batch_size)
    pbar = tqdm(range(0, len(df), batch_size), total=n_batches, desc="Embedding", unit="batch")
    for batch_start in pbar:
        batch_df = df.iloc[batch_start : batch_start + batch_size]
        patches = []
        batch_valid = []

        for idx, row in batch_df.iterrows():
            zarr_path = row["zarr_path"]
            cy, cx = int(row["object_y_fov"]), int(row["object_x_fov"])

            if zarr_path != current_fov_path:
                current_fov = store[zarr_path]["0"][0, :, 0]
                current_fov_path = zarr_path

            patch = crop_cell(current_fov, cy, cx, raw_half, channels=channel_indices)
            if patch is None:
                skipped_border += 1
                continue

            patches.append(patch)
            batch_valid.append(idx)

        if not patches:
            continue

        batch_tensor = torch.from_numpy(np.stack(patches)).float()
        if raw_crop_size != patch_size:
            batch_tensor = F.interpolate(
                batch_tensor, size=(patch_size, patch_size), mode="bilinear", align_corners=False
            )

        # Per-sample z-score: zero mean, unit std
        mean = batch_tensor.flatten(1).mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        std = batch_tensor.flatten(1).std(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)
        batch_tensor = (batch_tensor - mean) / std

        batch_tensor = batch_tensor.unsqueeze(2).to(device)

        with torch.inference_mode():
            features, _ = model(batch_tensor)

        all_embeddings.append(features.cpu().numpy())
        valid_indices.extend(batch_valid)
        pbar.set_postfix(cells=len(valid_indices), skipped=skipped_border)

    if skipped_border > 0:
        logger.warning(f"Skipped {skipped_border} cells too close to FOV border")

    # --- Write cell-level anndata ---
    embeddings = np.concatenate(all_embeddings, axis=0)
    valid_df = df.iloc[valid_indices].reset_index(drop=True)
    logger.info(f"Embeddings: {embeddings.shape}")

    pd.options.future.infer_string = False
    obs = valid_df.copy()
    for col in obs.select_dtypes(include=["string", "string[pyarrow]"]).columns:
        obs[col] = obs[col].astype(object)
    obs.index = obs["object_uuid"].astype(str)

    cell_adata = ad.AnnData(X=embeddings.astype(np.float32), obs=obs)
    cell_adata.write_zarr(output_path)
    logger.info(f"Wrote {output_path}: {cell_adata.n_obs} cells x {cell_adata.n_vars} dims")


if __name__ == "__main__":
    main()
