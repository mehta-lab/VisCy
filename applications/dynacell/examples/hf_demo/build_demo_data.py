"""Build the demo-data zips for biohub/dynacell-demo-data from the public release.

One held-out A549 test FOV per marker (``fov0006``, the FOV of the release reviewer
demo), mock condition, read from the release's ``data/biohub-a549/test/{MARKER}_mock.ozx``:

- T: 5 of the 10 timepoints (every other one, 5-21 hpi)
- C: Phase3D, Brightfield, target fluorescence (index 2, as the Space expects)
- Z: the central 32 of 48 slices, the Z window FNet3D was trained on
- YX: the central 512 x 512 of the 640 x 960 FOV

Each store is written as OME-Zarr v0.5 and zipped (stored, chunks are already
compressed) to ``{out_dir}/{MARKER}_mock.zarr.zip`` with ``{MARKER}_mock.zarr/`` at the
zip root. Run on HPC:

    uv run python applications/dynacell/examples/hf_demo/build_demo_data.py OUT_DIR
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
from iohub import open_ome_zarr
from iohub.ngff.models import TransformationMeta

RELEASE_TEST = Path("/hpc/projects/virtual_staining/dynacell_v1/data/biohub-a549/test")
MARKERS = ("CAAX", "H2B", "SEC61B", "TOMM20")
FOV = "0/0/fov0006"
TIMEPOINTS = [0, 2, 4, 6, 8]
Z_SIZE = 32
YX_SIZE = 512


def build_store(marker: str, out_dir: Path) -> Path:
    """Crop one marker's test FOV into ``{marker}_mock.zarr``; return the store path."""
    src_path = RELEASE_TEST / f"{marker}_mock.ozx"
    with open_ome_zarr(src_path, mode="r") as plate:
        src = plate[FOV]
        _, _, n_z, n_y, n_x = src.data.shape
        z0, y0, x0 = (n_z - Z_SIZE) // 2, (n_y - YX_SIZE) // 2, (n_x - YX_SIZE) // 2
        data = np.stack([src.data[t, :, z0 : z0 + Z_SIZE, y0 : y0 + YX_SIZE, x0 : x0 + YX_SIZE] for t in TIMEPOINTS])
        channel_names = list(src.channel_names)
        scale = src.scale
        hpi = [src.zattrs["hpi_values"][t] for t in TIMEPOINTS]
    # Released ER/mito targets are raw camera counts; deconvolved ones go negative.
    neg = float((data[:, 2] < 0).mean())
    if neg > 0:
        raise ValueError(f"{marker}: {neg:.4%} negative target pixels, expected raw (0%)")

    store = out_dir / f"{marker}_mock.zarr"
    with open_ome_zarr(store, layout="hcs", mode="w-", channel_names=channel_names, version="0.5") as out:
        pos = out.create_position("0", "0", "fov0006")
        pos.create_image(
            "0",
            data,
            chunks=(1, 1, Z_SIZE, YX_SIZE, YX_SIZE),
            transform=[TransformationMeta(type="scale", scale=scale)],
        )
        pos.zattrs["dynacell_demo"] = {
            "source": f"s3://dynacell/v1/data/biohub-a549/test/{marker}_mock.ozx",
            "position": FOV,
            "timepoints": TIMEPOINTS,
            "hpi_values": hpi,
            "zyx_offset": [z0, y0, x0],
            "zyx_size": [Z_SIZE, YX_SIZE, YX_SIZE],
        }
    print(f"{marker}: {data.shape} {data.dtype}, zyx offset {(z0, y0, x0)}, channels {channel_names}")
    return store


def zip_store(store: Path) -> Path:
    """Zip ``store`` (stored) with the store directory at the zip root."""
    zip_path = store.with_name(store.name + ".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in sorted(store.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(store.parent))
    return zip_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for marker in MARKERS:
        store = build_store(marker, args.out_dir)
        zip_path = zip_store(store)
        shutil.rmtree(store)
        print(f"  -> {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
