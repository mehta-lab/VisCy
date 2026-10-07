"""OME-Zarr metadata validation and atomic, resumable frame artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dynaclr.phenotyping.algorithms import FrameSkip


@dataclass(frozen=True)
class Position:
    """Describe native-resolution axes, coordinates and channel identities."""

    path: Path
    attrs: dict
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    scale: tuple[float, ...]
    translation: tuple[float, ...]
    units: tuple[str | None, ...]
    channels: tuple[str, ...]
    array_path: str

    @property
    def pixel_size_yx_um(self) -> tuple[float, float]:
        """Return XY pixel spacing after validating micrometer units."""
        for axis in (3, 4):
            if self.units[axis] not in ("micrometer", "micrometre", "um", "µm"):
                raise FrameSkip("missing_or_non_micrometer_xy_units")
        return self.scale[3], self.scale[4]

    def open(self):
        """Open the native array for read-only frame access."""
        import zarr

        return zarr.open_group(str(self.path), mode="r")[self.array_path]


def attrs_at(path: Path) -> dict:
    """Read group attributes from Zarr v2 or v3 metadata without image chunks."""
    if (path / "zarr.json").exists():
        return json.loads((path / "zarr.json").read_text()).get("attributes", {})
    if (path / ".zattrs").exists():
        return json.loads((path / ".zattrs").read_text())
    raise FileNotFoundError(f"Missing group metadata: {path}")


def positions(root: str | Path) -> list[str]:
    """List exact row/well/FOV identities without traversing image chunks."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = []
    for row in root.iterdir():
        if not row.is_dir() or row.name.startswith("."):
            continue
        for well in row.iterdir():
            if not well.is_dir():
                continue
            for fov in well.iterdir():
                if fov.is_dir() and ((fov / "zarr.json").exists() or (fov / ".zattrs").exists()):
                    result.append(f"{row.name}/{well.name}/{fov.name}")
    return sorted(result)


def read_position(path: str | Path) -> Position:
    """Read and validate native-resolution OME coordinate metadata."""
    path = Path(path)
    attrs = attrs_at(path)
    ome = attrs.get("ome", attrs)
    multiscales = ome.get("multiscales", [])
    if len(multiscales) != 1:
        raise FrameSkip("missing_or_ambiguous_multiscales")
    multiscale = multiscales[0]
    axes = multiscale.get("axes", [])
    names = tuple(a["name"].lower() if isinstance(a, dict) else a.lower() for a in axes)
    if names != ("t", "c", "z", "y", "x"):
        raise FrameSkip("unsupported_axis_order")
    units = tuple(a.get("unit") if isinstance(a, dict) else None for a in axes)
    datasets = multiscale.get("datasets", [])
    if not datasets:
        raise FrameSkip("missing_resolution_dataset")
    dataset = datasets[0]
    array_path = dataset["path"]
    if array_path != "0":
        raise FrameSkip("native_resolution_path_not_zero")
    metadata_path = path / array_path / "zarr.json"
    metadata = (
        json.loads(metadata_path.read_text())
        if metadata_path.exists()
        else json.loads((path / array_path / ".zarray").read_text())
    )
    shape = tuple(metadata["shape"])
    if len(shape) != 5 or any(n <= 0 for n in shape):
        raise FrameSkip("invalid_image_shape")
    dims = metadata.get("dimension_names")
    if dims is not None and tuple(a.lower() for a in dims) != names:
        raise FrameSkip("array_metadata_axis_mismatch")
    transformations = dataset.get("coordinateTransformations", []) + multiscale.get("coordinateTransformations", [])
    scale = np.ones(5)
    translation = np.zeros(5)
    have_scale = False
    for transform in transformations:
        if transform.get("type") == "scale":
            values = np.array(transform["scale"], dtype=float)
            if values.shape != (5,) or not np.isfinite(values).all() or (values <= 0).any():
                raise FrameSkip("invalid_coordinate_scale")
            scale *= values
            translation *= values
            have_scale = True
        elif transform.get("type") == "translation":
            values = np.array(transform["translation"], dtype=float)
            if values.shape != (5,) or not np.isfinite(values).all():
                raise FrameSkip("invalid_coordinate_translation")
            translation += values
        elif transform.get("type") != "identity":
            raise FrameSkip("unsupported_coordinate_transform")
    if not have_scale:
        raise FrameSkip("missing_coordinate_scale")
    channels = tuple(ch["label"] for ch in ome.get("omero", {}).get("channels", []))
    if len(channels) != shape[1] or len(set(channels)) != len(channels):
        raise FrameSkip("missing_or_duplicate_channel_names")
    return Position(path, attrs, shape, names, tuple(scale), tuple(translation), units, channels, array_path)


def validate_alignment(source: Position, focus: Position, tracking: Position) -> None:
    """Verify matching spatial grids and trajectory lengths before pixel checks."""
    if source.shape[0] != focus.shape[0] or source.shape[2:] != focus.shape[2:]:
        raise FrameSkip("focus_source_coordinate_shape_mismatch")
    if source.shape[0] != tracking.shape[0] or source.shape[3:] != tracking.shape[3:]:
        raise FrameSkip("tracking_coordinate_shape_mismatch")
    if tracking.shape[1:3] != (1, 1):
        raise FrameSkip("tracking_not_single_channel_2d")
    for other, axes in ((focus, (2, 3, 4)), (tracking, (3, 4))):
        if any(source.units[a] != other.units[a] for a in axes):
            raise FrameSkip("coordinate_unit_mismatch")
        if not np.allclose(np.array(source.scale)[list(axes)], np.array(other.scale)[list(axes)], rtol=1e-8, atol=0):
            raise FrameSkip("coordinate_scale_mismatch")
        if not np.allclose(
            np.array(source.translation)[list(axes)], np.array(other.translation)[list(axes)], rtol=0, atol=1e-8
        ):
            raise FrameSkip("coordinate_origin_mismatch")
    # Prepared/tracking have legacy T scale=1. Source/focus phase pixel equality
    # and CSV time/centroid checks establish index identity before any measurement.
    source.pixel_size_yx_um


def find_track_csv(root: str | Path, fov: str) -> Path:
    """Find the unique tracking CSV belonging to a FOV."""
    candidates = sorted((Path(root) / fov).glob("*.csv"))
    if len(candidates) != 1:
        raise FrameSkip("missing_or_ambiguous_track_csv")
    return candidates[0]


def atomic_json(path: Path, data: dict) -> None:
    """Replace a JSON artifact atomically after serializing finite values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_parquet(path: Path, data: pd.DataFrame) -> None:
    """Replace a Parquet artifact atomically after writing its complete table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    data.to_parquet(tmp, index=False)
    os.replace(tmp, path)


@contextmanager
def lock(path: Path, blocking: bool = True):
    """Hold a process advisory lock for output initialization or frame ownership."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(stream, flags)
        except BlockingIOError:
            raise RuntimeError(f"Another worker owns {path}") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def ensure_labels(output: Path, fov: str, source: Position):
    """Initialize native label arrays and HCS metadata under one shared lock."""
    import zarr

    with lock(output / "locks" / "label_metadata.lock"):
        root = zarr.open_group(str(output / "segmentations.ome.zarr"), mode="a", zarr_format=3)
        root.attrs["phenotyping"] = {
            "source": str(source.path.parents[2]),
            "label_ids": "frame_local_cells_and_track_id_nuclei",
        }
        group = root.require_group(fov)
        # Preserve source HCS row/column identities while advertising only groups
        # already initialized. This critical section also owns row/well creation.
        import copy

        source_ome = attrs_at(source.path.parents[2]).get("ome", {})
        if "plate" in source_ome:
            plate = copy.deepcopy(source_ome["plate"])
            plate["wells"] = [well for well in plate.get("wells", []) if well["path"] in root]
            root.attrs["ome"] = {"version": "0.5", "plate": plate}
        well_path = "/".join(fov.split("/")[:2])
        well = root[well_path]
        well_ome = attrs_at(source.path.parent).get("ome", {})
        if "well" in well_ome:
            metadata = copy.deepcopy(well_ome["well"])
            metadata["images"] = [image for image in metadata.get("images", []) if image["path"] in well]
        else:
            metadata = {"images": [{"path": name} for name in well.group_keys()]}
        well.attrs["ome"] = {"version": "0.5", "well": metadata}
        shape = (source.shape[0], 1, 1, *source.shape[3:])
        axes = [
            {"name": "t", "type": "time", "unit": source.units[0] or "second"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": source.units[2]},
            {"name": "y", "type": "space", "unit": source.units[3]},
            {"name": "x", "type": "space", "unit": source.units[4]},
        ]
        scale = list(source.scale)
        scale[1] = 1.0
        origin = list(source.translation)
        origin[1] = 0.0
        # z=0 is a projection, not a single acquired plane. Bounds are per-frame status.
        origin[2] = 0.0
        transforms = [{"type": "scale", "scale": scale}, {"type": "translation", "translation": origin}]
        multiscales = [{"axes": axes, "datasets": [{"path": "0", "coordinateTransformations": transforms}]}]
        ome = {"version": "0.5", "multiscales": multiscales, "omero": {"channels": [{"label": "cell_labels"}]}}
        group.attrs["ome"] = ome
        group.attrs["projection"] = {"method": "maximum", "focus_channel": "Phase3D", "offsets": [-1, 3]}
        arrays = {}
        for name, target in (
            ("cells", group),
            ("cytoplasm", group.require_group("labels/cytoplasm")),
            ("nuclei", group.require_group("labels/nuclei")),
        ):
            if name != "cells":
                target.attrs["ome"] = {
                    "version": "0.5",
                    "multiscales": multiscales,
                    "image-label": {"version": "0.5", "source": {"image": "../../"}},
                }
            if "0" not in target:
                target.create_array(
                    "0",
                    shape=shape,
                    chunks=(1, 1, 1, min(256, shape[3]), min(256, shape[4])),
                    dtype="uint32",
                    fill_value=0,
                    dimension_names=("t", "c", "z", "y", "x"),
                )
            array = target["0"]
            if tuple(array.shape) != shape or np.dtype(array.dtype) != np.dtype("uint32"):
                raise RuntimeError("Existing output label dimensions/dtype differ")
            arrays[name] = array
        group["labels"].attrs["ome"] = {"version": "0.5", "labels": ["cytoplasm", "nuclei"]}
        return arrays


def stable_hash(data: Any) -> str:
    """Hash a JSON-serializable value using a canonical representation."""
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_identity(config) -> dict:
    """Hash source metadata and tracking CSVs for reproducible input identity."""
    result = {}
    for kind in ("source", "focus_source", "tracks"):
        root = Path(getattr(config, kind))
        digest = hashlib.sha256()
        for fov in positions(root):
            for rel in ("zarr.json", ".zattrs", "0/zarr.json", "0/.zarray"):
                file = root / fov / rel
                if file.exists():
                    digest.update(f"{fov}/{rel}".encode())
                    digest.update(file.read_bytes())
            if kind == "tracks":
                for file in sorted((root / fov).glob("*.csv")):
                    digest.update(file.name.encode())
                    digest.update(file.read_bytes())
        result[kind] = {
            "path": str(root.resolve()),
            "metadata_and_csv_sha256": digest.hexdigest(),
            "pixel_chunks": "read_only_source_assumed_immutable; consumed_frame_hashes_in_status",
        }
    return result


def code_identity() -> dict:
    """Record scoped source hashes, Git identity and installed package versions."""
    package = Path(__file__).parent
    files = sorted(package.glob("*.py"))
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        digest.update(file.read_bytes())

    def git(*args):
        result = subprocess.run(["git", "-C", str(package), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    relative = "applications/dynaclr/src/dynaclr/phenotyping"
    diff = git("diff", "HEAD", "--", relative)
    versions = {}
    for name in (
        "cellpose",
        "cp-measure",
        "dinov3",
        "torch",
        "numpy",
        "scipy",
        "scikit-image",
        "pandas",
        "pyarrow",
        "zarr",
        "anndata",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    # Editable/direct-URL DINO provenance matters in addition to its version.
    try:
        direct = importlib.metadata.distribution("dinov3").read_text("direct_url.json")
        versions["dinov3_direct_url"] = json.loads(direct) if direct else None
    except importlib.metadata.PackageNotFoundError:
        pass
    return {
        "source_sha256": digest.hexdigest(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "scoped_dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "versions": versions,
        "python": sys.version.split()[0],
    }


def snapshot_code(output: Path, config_path: Path | None) -> None:
    """Archive the owned source files and original recipe before an attempt starts."""
    import shutil

    target = output / "provenance" / "source"
    target.mkdir(parents=True, exist_ok=True)
    for file in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(file, target / file.name)
    if config_path is not None:
        shutil.copyfile(config_path, output / "provenance" / "input_config.yml")


def assert_compute_node() -> None:
    """Reject image processing or summary rendering on a login node."""
    import socket

    hostname = socket.gethostname().split(".")[0].lower()
    if hostname.startswith("login") or hostname in ("bruno", "fry1", "fry2"):
        raise RuntimeError("Image processing and summary plotting must run on an allocated compute node")
