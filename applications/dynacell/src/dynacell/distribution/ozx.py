"""Pack assembled OME-Zarr stores into RFC-9 ``.ozx`` archives.

Wraps :func:`iohub.core.ozx.pack_ozx` (1:1 zip-pack of a directory zarr
store, RFC-9 compliant — see iohub PR #408) with two extras the bare
API does not provide:

- **Sample mode**: a subset writer that copies a bounded number of FOVs
  and timepoints into a tmp zarr before packing — needed for the
  NeurIPS ``<4 GB`` reviewer sample.
- **Manifest emission**: writes per-dataset MANIFEST.json with sha256 +
  size + ozx_version per packed file. Stream B's output is Stream A's
  input (Croissant ``cr:FileObject.sha256`` populated from this).
"""

import dataclasses
import datetime
import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from iohub.core.ozx import (
    OZX_EXTENSION,
    is_ozx_path,
    pack_ozx,
    read_ozx_version,
)

from dynacell.data import get_manifest

PackMode = Literal["all", "sample"]

__all__ = [
    "OZX_EXTENSION",
    "PackMode",
    "PackResult",
    "is_ozx_path",
    "pack_dataset",
    "read_ozx_version",
]

_BUF_SIZE = 1 << 20  # 1 MiB chunks for sha256 streaming.

# `StoreLocations` fields that are data splits rather than auxiliary artifacts.
# A requested core split with no store is an error; the auxiliary fields
# (`cell_segmentation`, `gt_cache_dir`) are legitimately unset per target.
_CORE_SPLITS: frozenset[str] = frozenset({"train", "test"})


@dataclasses.dataclass(frozen=True)
class PackResult:
    """Single ``(target, split)`` packed OZX result.

    Attributes
    ----------
    dataset
        Registered dataset name (e.g. ``"aics-hipsc"``).
    target
        Target key within the manifest (e.g. ``"sec61b"``).
    split
        ``"train"`` or ``"test"``.
    src_zarr_path
        Source OME-Zarr directory (the assembled store).
    dst_ozx_path
        Output ``.ozx`` archive.
    bytes
        Size of the resulting archive in bytes.
    sha256
        Hex digest of the OZX file.
    ozx_version
        OME-NGFF version stamped in the OZX archive comment.
    """

    dataset: str
    target: str
    split: str
    src_zarr_path: Path
    dst_ozx_path: Path
    bytes: int
    sha256: str
    ozx_version: str | None


def pack_dataset(
    name: str,
    *,
    output_root: Path,
    mode: PackMode = "all",
    fov_limit: int | None = None,
    t_limit: int | None = None,
    overwrite: bool = False,
    targets: list[str] | None = None,
    splits: list[str] | None = None,
) -> list[PackResult]:
    """Pack one registered dataset's assembled zarrs into OZX archives.

    Parameters
    ----------
    name
        Registered dataset name (must resolve via
        :func:`dynacell.data.get_manifest`).
    output_root
        Top-level output directory; OZX files land at
        ``<output_root>/<dataset>/<split>/<store-name>.ozx``. The path
        layout mirrors the source store layout from the manifest.
    mode
        ``"all"`` packs the full source zarr 1:1. ``"sample"`` writes
        a subset zarr (limited by ``fov_limit`` and ``t_limit``) to a
        tmp directory then packs that — ``pack_ozx`` itself does not
        sub-select.
    fov_limit, t_limit
        Per-axis limits in sample mode. Ignored in ``"all"`` mode.
    overwrite
        Replace any existing destination ``.ozx``.
    targets, splits
        Optional filters. Default: every target × ``{train, test}``.

    Returns
    -------
    list[PackResult]
        One entry per unique ``(physical_source, split)`` pair. Targets
        sharing a physical source store (e.g. aics-hipsc nucleus +
        membrane both reference ``cell.zarr``) are de-duplicated to a
        single PackResult.
    """
    manifest = get_manifest(name)
    requested_targets = list(targets) if targets is not None else list(manifest.targets)
    requested_splits = list(splits) if splits is not None else ["train", "test"]

    # Validate targets + splits up front so a typo (e.g. "tran" instead
    # of "train") fails loudly rather than silently producing zero
    # outputs because getattr(...None) would otherwise skip the missing
    # attribute. Split names must match a `StoreLocations` field.
    unknown_targets = [t for t in requested_targets if t not in manifest.targets]
    if unknown_targets:
        raise ValueError(
            f"Unknown targets {unknown_targets!r} for dataset {name!r}; available: {sorted(manifest.targets)}"
        )
    sample_target = next(iter(manifest.targets.values()))
    valid_split_fields = set(type(sample_target.stores).model_fields)
    unknown_splits = [s for s in requested_splits if s not in valid_split_fields]
    if unknown_splits:
        raise ValueError(f"Unknown splits {unknown_splits!r}; available: {sorted(valid_split_fields)}")

    # `train` / `test` are the data splits a pack request means literally, and
    # `train` is optional on evaluation-only manifests (hek-mantis-*). Without
    # this check the per-split loop below would hit its `src is None` branch --
    # written for the auxiliary fields -- and silently emit a manifest missing
    # the split the caller asked for (the CLI requests train,test by default).
    missing_core = [
        f"{target_key}/{split}"
        for split in requested_splits
        if split in _CORE_SPLITS
        for target_key in requested_targets
        if getattr(manifest.targets[target_key].stores, split) is None
    ]
    if missing_core:
        raise ValueError(
            f"Dataset {name!r} has no store for requested split(s) {missing_core!r}. "
            "Evaluation-only datasets define no train store — pass --splits test."
        )

    seen_sources: dict[Path, PackResult] = {}
    results: list[PackResult] = []

    for target_key in requested_targets:
        target = manifest.targets[target_key]
        for split in requested_splits:
            src = getattr(target.stores, split, None)
            if src is None:
                # The split is a valid StoreLocations field (validated
                # above) but this target left it unset (e.g.
                # `cell_segmentation` is optional); skip silently.
                continue
            src_path = Path(src)
            if not src_path.exists():
                raise FileNotFoundError(f"Source store missing: {src_path} ({name}/{target_key}/{split})")
            if src_path in seen_sources:
                # Dedup: shared store across targets emits one OZX.
                continue
            # mode="sample" MUST NOT collide with the full archive. The
            # `_sample` suffix used to live only on the throwaway tmp zarr, so
            # `pack <ds>` followed by `sample <ds> --overwrite` into the same
            # --output-root replaced a multi-GB release archive with a 2-FOV
            # reviewer subset -- and write_pack_manifest then recorded the
            # sample's sha256/bytes as the dataset's.
            stem = f"{src_path.stem}_sample" if mode == "sample" else src_path.stem
            dst_path = output_root / name / split / f"{stem}.ozx"
            result = _pack_one(
                dataset=name,
                target=target_key,
                split=split,
                src_path=src_path,
                dst_path=dst_path,
                mode=mode,
                fov_limit=fov_limit,
                t_limit=t_limit,
                overwrite=overwrite,
            )
            seen_sources[src_path] = result
            results.append(result)
    return results


def _pack_one(
    *,
    dataset: str,
    target: str,
    split: str,
    src_path: Path,
    dst_path: Path,
    mode: PackMode,
    fov_limit: int | None,
    t_limit: int | None,
    overwrite: bool,
) -> PackResult:
    """Pack a single source zarr into one OZX archive."""
    if dst_path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst_path}; pass overwrite=True")
        dst_path.unlink()
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "sample":
        if fov_limit is None or t_limit is None:
            raise ValueError("sample mode requires both fov_limit and t_limit")
        with _subset_zarr(src_path, fov_limit=fov_limit, t_limit=t_limit) as subset_path:
            pack_ozx(subset_path, dst_path)
    else:
        pack_ozx(src_path, dst_path)

    return PackResult(
        dataset=dataset,
        target=target,
        split=split,
        src_zarr_path=src_path,
        dst_ozx_path=dst_path,
        bytes=dst_path.stat().st_size,
        sha256=_sha256_file(dst_path),
        ozx_version=read_ozx_version(dst_path),
    )


def _sha256_file(path: Path) -> str:
    """Stream the file through sha256 in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_BUF_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    """UTC timestamp without microseconds for MANIFEST.json determinism debug."""
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


class _subset_zarr:
    """Context manager that writes a subset zarr to a tmp dir for sample-mode pack.

    Iohub's ``pack_ozx`` is a 1:1 archive; this helper handles the
    subset-extraction step. Only HCS plate sources are supported (the
    only layout dynacell-paper produces).
    """

    def __init__(
        self,
        src_path: Path,
        *,
        fov_limit: int,
        t_limit: int,
    ) -> None:
        self.src_path = src_path
        self.fov_limit = fov_limit
        self.t_limit = t_limit
        self.tmp_dir: Path | None = None

    def __enter__(self) -> Path:
        from iohub.ngff import open_ome_zarr

        self.tmp_dir = Path(tempfile.mkdtemp(prefix="dynacell-ozx-sample-"))
        dst_zarr = self.tmp_dir / f"{self.src_path.stem}_sample.zarr"

        with open_ome_zarr(self.src_path, mode="r", layout="hcs") as src:
            channel_names = src.channel_names
            with open_ome_zarr(
                dst_zarr,
                mode="w-",
                layout="hcs",
                channel_names=channel_names,
            ) as dst:
                for i, (pos_name, src_pos) in enumerate(src.positions()):
                    if i >= self.fov_limit:
                        break
                    src_arr = src_pos["0"]
                    t_take = min(self.t_limit, src_arr.shape[0])
                    sub = src_arr[:t_take, ...]
                    dst_pos = dst.create_position(*pos_name.split("/"))
                    dst_pos.create_image("0", sub, chunks=src_arr.chunks)
        return dst_zarr

    def __exit__(self, *exc: object) -> None:
        if self.tmp_dir is not None and self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
