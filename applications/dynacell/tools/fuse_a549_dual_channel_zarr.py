"""Fuse A549 CAAX + H2B pool-train zarrs into a single 3-channel dual store.

The two pooled-per-marker train zarrs (CAAX_all.zarr with channels
[Phase3D, Brightfield, Membrane] and H2B_all.zarr with channels
[Phase3D, Brightfield, Nuclei]) are channel-extracted views of the same
source acquisition (verified plate_id=2026_03_26_A549_CAAX_H2B_DENV_ZIKV;
30/30 train + 12/12 test source_position overlap per the provenance sidecars).

This script joins them by `source_position` (the real plate position from
provenance), then writes a new OME-Zarr v3 store with channels
[Phase3D, Nuclei, Membrane] suitable for Track C dual nucleus+membrane
fine-tuning (`vscyto3d_cytolandft` / `vscyto3d_infectionft_dynacellft` on
a549_mantis).

A Phase3D byte-equality gate (np.allclose) runs per position to catch any
divergence from the source_position join — Phase3D *should* be identical
between the two channel-extracted views, so any mismatch indicates the join
is wrong and the run aborts.

Usage::

    uv run python applications/dynacell/tools/fuse_a549_dual_channel_zarr.py \
        --caax /hpc/projects/virtual_staining/training/dynacell/a549/mantis_v1/train/CAAX_all.zarr \
        --h2b  /hpc/projects/virtual_staining/training/dynacell/a549/mantis_v1/train/H2B_all.zarr \
        --out  /hpc/projects/comp.micro/virtual_staining/datasets/dynacell/a549_mantis_dual_nucl_memb_all.zarr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from iohub.ngff import TransformationMeta, open_ome_zarr

# Channel name layout for the OUTPUT dual store. Order matches what Track C
# train_sets/a549_mantis_dual.yml advertises: source first, then target
# channels in (Nuclei, Membrane) order (matches the cytoland public ckpt's
# decoder channel order; Track B/C predict eval reads `Nuclei_prediction`
# from channel 0 and `Membrane_prediction` from channel 1).
OUT_CHANNELS = ["Phase3D", "Nuclei", "Membrane"]


def _provenance_path(zarr_path: Path) -> Path:
    """Return the colocated provenance.json sidecar path for a pool zarr."""
    return zarr_path.with_suffix(".provenance.json")


def _build_source_position_map(provenance: dict) -> dict[str, str]:
    """Return {source_position: pool_position_key} for one pool zarr's provenance."""
    out: dict[str, str] = {}
    for pool_key, entry in provenance["positions"].items():
        src = entry["source_position"]
        if src in out:
            raise ValueError(
                f"duplicate source_position {src!r} in provenance: maps to both {out[src]!r} and {pool_key!r}"
            )
        out[src] = pool_key
    return out


def main() -> int:
    """Fuse CAAX + H2B pool zarrs into one dual nucl+memb store."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caax", required=True, type=Path)
    parser.add_argument("--h2b", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--allclose-atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for the per-position Phase3D byte-equality gate.",
    )
    parser.add_argument(
        "--allclose-rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for the per-position Phase3D byte-equality gate.",
    )
    args = parser.parse_args()

    caax_provenance = json.loads(_provenance_path(args.caax).read_text())
    h2b_provenance = json.loads(_provenance_path(args.h2b).read_text())

    if caax_provenance["target"].lower() != "caax":
        raise ValueError(f"--caax provenance target is {caax_provenance['target']!r}, expected 'caax'")
    if h2b_provenance["target"].lower() != "h2b":
        raise ValueError(f"--h2b provenance target is {h2b_provenance['target']!r}, expected 'h2b'")
    if caax_provenance["condition"] != h2b_provenance["condition"]:
        raise ValueError(
            f"condition mismatch: caax={caax_provenance['condition']!r} h2b={h2b_provenance['condition']!r}"
        )
    if caax_provenance["split"] != h2b_provenance["split"]:
        raise ValueError(f"split mismatch: caax={caax_provenance['split']!r} h2b={h2b_provenance['split']!r}")

    caax_map = _build_source_position_map(caax_provenance)
    h2b_map = _build_source_position_map(h2b_provenance)
    shared = sorted(set(caax_map) & set(h2b_map))
    only_caax = sorted(set(caax_map) - set(h2b_map))
    only_h2b = sorted(set(h2b_map) - set(caax_map))
    if only_caax or only_h2b:
        raise ValueError(f"source_position sets diverge: only_caax={only_caax!r} only_h2b={only_h2b!r}")
    if not shared:
        raise ValueError("no shared source_positions between CAAX and H2B provenance")

    print(f"[fuse] shared positions: {len(shared)}")
    print(f"[fuse] caax store: {args.caax}")
    print(f"[fuse] h2b store:  {args.h2b}")
    print(f"[fuse] output:     {args.out}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with (
        open_ome_zarr(args.caax, mode="r") as caax_plate,
        open_ome_zarr(args.h2b, mode="r") as h2b_plate,
        open_ome_zarr(
            args.out,
            layout="hcs",
            mode="w-",
            channel_names=OUT_CHANNELS,
        ) as out_plate,
    ):
        caax_phase_idx = caax_plate.channel_names.index("Phase3D")
        caax_memb_idx = caax_plate.channel_names.index("Membrane")
        h2b_phase_idx = h2b_plate.channel_names.index("Phase3D")
        h2b_nucl_idx = h2b_plate.channel_names.index("Nuclei")

        for src_pos in shared:
            caax_key = caax_map[src_pos]
            h2b_key = h2b_map[src_pos]
            caax_pos = caax_plate[caax_key]
            h2b_pos = h2b_plate[h2b_key]
            if caax_pos.data.shape != h2b_pos.data.shape:
                raise ValueError(
                    f"shape mismatch for source_position={src_pos!r}: "
                    f"caax={caax_pos.data.shape} h2b={h2b_pos.data.shape}"
                )
            caax_arr = np.asarray(caax_pos.data)
            h2b_arr = np.asarray(h2b_pos.data)
            phase_caax = caax_arr[:, caax_phase_idx]
            phase_h2b = h2b_arr[:, h2b_phase_idx]
            if not np.allclose(phase_caax, phase_h2b, atol=args.allclose_atol, rtol=args.allclose_rtol):
                diff = np.abs(phase_caax - phase_h2b).max()
                raise ValueError(
                    f"Phase3D mismatch for source_position={src_pos!r}: "
                    f"max abs diff = {diff!r}. source_position join may be wrong."
                )
            membrane = caax_arr[:, caax_memb_idx]
            nuclei = h2b_arr[:, h2b_nucl_idx]
            # Stack as (T, 3, Z, Y, X) matching OUT_CHANNELS order.
            fused = np.stack([phase_caax, nuclei, membrane], axis=1)
            if np.isnan(fused).any():
                raise ValueError(f"NaN encountered in fused array for source_position={src_pos!r}")
            row_name, col_name, pos_name = caax_key.split("/")
            new_pos = out_plate.create_position(row_name, col_name, pos_name)
            scale = list(caax_pos.scale)
            transform = [TransformationMeta(type="scale", scale=scale)]
            chunks = (1, 1) + tuple(fused.shape[-3:])
            new_pos.create_image(
                "0",
                fused,
                chunks=chunks,
                transform=transform,
            )
            print(f"[fuse] wrote {caax_key} (source_position={src_pos})")

    print(f"[fuse] done: {len(shared)} positions written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
