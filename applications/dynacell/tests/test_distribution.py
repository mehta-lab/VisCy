"""Tests for dynacell.distribution OZX packaging.

Pack a tiny synthetic OME-Zarr fixture, verify the resulting ``.ozx``
round-trips back to a directory zarr, and check the MANIFEST.json
shape. Real-data E2E tests live behind an opt-in environment variable
so the suite stays fast off-HPC.
"""

import dataclasses
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from iohub.core.ozx import (
    OZX_EXTENSION,
    is_ozx_path,
    pack_ozx,
    read_ozx_version,
)
from iohub.ngff import open_ome_zarr


def _make_fixture_zarr(path: Path, n_pos: int, t: int) -> None:
    """Write a tiny HCS OME-Zarr with ``n_pos`` positions × ``t`` timepoints."""
    with open_ome_zarr(
        path,
        mode="w-",
        layout="hcs",
        channel_names=["Phase3D", "Brightfield"],
    ) as plate:
        for i in range(n_pos):
            row, col, fov = "B", "1", f"{i:06d}"
            position = plate.create_position(row, col, fov)
            data = np.zeros((t, 2, 1, 8, 8), dtype=np.float32)
            data[..., 0, 0] = float(i)  # discriminator value
            position.create_image("0", data)


class TestUpstreamOzx:
    """Sanity smoke that the imported iohub.core.ozx surface works.

    Round-trip + extension constant covered by iohub's own tests; this
    class only verifies the project consumes upstream correctly so a
    silent regression after a future iohub bump is caught here.
    """

    def test_extension_constant(self):
        """Imported OZX_EXTENSION matches the RFC-9 spec."""
        assert OZX_EXTENSION == ".ozx"

    def test_is_ozx_path(self):
        """is_ozx_path is exported and case-sensitive (matches iohub)."""
        assert is_ozx_path("foo.ozx")
        # iohub is deliberately case-sensitive, mirroring the `.zarr`
        # precedent, so an uppercase suffix is not recognized.
        assert not is_ozx_path(Path("a/b/foo.OZX"))
        assert not is_ozx_path("foo.zarr")

    def test_pack_round_trip_small(self, tmp_path):
        """Pack a fixture and re-open it via iohub.open_ome_zarr."""
        src = tmp_path / "src.zarr"
        _make_fixture_zarr(src, n_pos=2, t=3)
        dst = tmp_path / "out.ozx"

        pack_ozx(src, dst)

        assert dst.exists()
        version = read_ozx_version(dst)
        assert version is not None

        with open_ome_zarr(dst, mode="r", layout="hcs") as plate:
            positions = list(plate.positions())
            assert len(positions) == 2
            for _, pos in positions:
                arr = pos["0"][:]
                assert arr.shape[0] == 3


class TestPackDataset:
    """Higher-level pack_dataset wrapper exercises the registry."""

    def test_pack_dataset_dedup(self, tmp_path, monkeypatch):
        """Nucleus + membrane share cell.zarr → one PackResult per physical source."""
        from dynacell.distribution import pack_dataset

        # Build a tiny fixture that mirrors aics-hipsc's shared-store pattern.
        train_zarr = tmp_path / "train_cell.zarr"
        test_zarr = tmp_path / "test_cell.zarr"
        _make_fixture_zarr(train_zarr, n_pos=1, t=1)
        _make_fixture_zarr(test_zarr, n_pos=1, t=1)

        # Stub manifest with 2 targets sharing the same physical store paths.
        from dynacell.data.manifests import (
            DatasetManifest,
            StoreLocations,
            TargetConfig,
            VoxelSpacing,
        )

        manifest = DatasetManifest(
            name="test-dataset",
            version="1",
            description="Test",
            cell_type="A549",
            imaging_modality="confocal",
            spacing=VoxelSpacing(z=1.0, y=1.0, x=1.0),
            channels={"source": "Phase3D"},
            targets={
                "nucleus": TargetConfig(
                    gene="H2B",
                    organelle="nucleus",
                    display_name="Nucleus",
                    target_channel="Nuclei",
                    stores=StoreLocations(train=train_zarr, test=test_zarr),
                    splits="splits/nucleus.yaml",
                ),
                "membrane": TargetConfig(
                    gene="CAAX",
                    organelle="membrane",
                    display_name="Membrane",
                    target_channel="Membrane",
                    stores=StoreLocations(train=train_zarr, test=test_zarr),
                    splits="splits/membrane.yaml",
                ),
            },
        )

        with patch(
            "dynacell.distribution.ozx.get_manifest",
            return_value=manifest,
        ):
            results = pack_dataset(
                "test-dataset",
                output_root=tmp_path / "out",
                mode="all",
            )

        assert len(results) == 2  # train + test, deduped to one each
        sources = {r.src_zarr_path for r in results}
        assert sources == {train_zarr, test_zarr}
        for r in results:
            assert r.dst_ozx_path.exists()
            assert r.bytes > 0
            assert len(r.sha256) == 64


class TestPackDatasetValidation:
    """pack_dataset rejects unknown targets / splits up front."""

    def _stub_manifest(self, tmp_path):
        from dynacell.data.manifests import (
            DatasetManifest,
            StoreLocations,
            TargetConfig,
            VoxelSpacing,
        )

        train_zarr = tmp_path / "train.zarr"
        test_zarr = tmp_path / "test.zarr"
        _make_fixture_zarr(train_zarr, n_pos=1, t=1)
        _make_fixture_zarr(test_zarr, n_pos=1, t=1)
        return DatasetManifest(
            name="t",
            version="1",
            description="t",
            cell_type="A549",
            imaging_modality="confocal",
            spacing=VoxelSpacing(z=1.0, y=1.0, x=1.0),
            channels={"source": "Phase3D"},
            targets={
                "x": TargetConfig(
                    gene="X",
                    organelle="x",
                    display_name="X",
                    target_channel="X",
                    stores=StoreLocations(train=train_zarr, test=test_zarr),
                    splits="splits/x.yaml",
                ),
            },
        )

    def test_unknown_target_raises(self, tmp_path):
        """Typo in --targets surfaces as ValueError, not silent zero output."""
        from dynacell.distribution import pack_dataset

        manifest = self._stub_manifest(tmp_path)
        with patch(
            "dynacell.distribution.ozx.get_manifest",
            return_value=manifest,
        ):
            with pytest.raises(ValueError, match="Unknown targets"):
                pack_dataset(
                    "t",
                    output_root=tmp_path / "out",
                    targets=["typo"],
                )

    def test_unknown_split_raises(self, tmp_path):
        """Typo in --splits surfaces as ValueError, not silent zero output."""
        from dynacell.distribution import pack_dataset

        manifest = self._stub_manifest(tmp_path)
        with patch(
            "dynacell.distribution.ozx.get_manifest",
            return_value=manifest,
        ):
            with pytest.raises(ValueError, match="Unknown splits"):
                pack_dataset(
                    "t",
                    output_root=tmp_path / "out",
                    splits=["tran"],  # typo for "train"
                )

    def test_missing_core_split_raises(self, tmp_path):
        """An evaluation-only manifest fails loudly on the default train,test request.

        ``StoreLocations.train`` is optional so evaluation-only datasets can ship a
        test store alone. Without an up-front check, the per-split loop's
        ``src is None`` branch (written for the auxiliary fields) would silently
        emit a manifest missing the requested train split.
        """
        from dynacell.distribution import pack_dataset

        manifest = self._stub_manifest(tmp_path)
        manifest.targets["x"].stores.train = None
        with patch(
            "dynacell.distribution.ozx.get_manifest",
            return_value=manifest,
        ):
            with pytest.raises(ValueError, match="no store for requested split"):
                pack_dataset("t", output_root=tmp_path / "out")  # defaults to train,test

    def test_missing_core_split_is_fine_when_not_requested(self, tmp_path):
        """Explicitly requesting only `test` on a train-less manifest still packs."""
        from dynacell.distribution import pack_dataset

        manifest = self._stub_manifest(tmp_path)
        manifest.targets["x"].stores.train = None
        with patch(
            "dynacell.distribution.ozx.get_manifest",
            return_value=manifest,
        ):
            results = pack_dataset("t", output_root=tmp_path / "out", splits=["test"])
        assert [r.split for r in results] == ["test"]


class TestSampleMode:
    """Sample mode writes a bounded subset zarr before packing."""

    def _manifest(self, train_zarr, test_zarr):
        from dynacell.data.manifests import (
            DatasetManifest,
            StoreLocations,
            TargetConfig,
            VoxelSpacing,
        )

        return DatasetManifest(
            name="test-dataset",
            version="1",
            description="Test",
            cell_type="A549",
            imaging_modality="confocal",
            spacing=VoxelSpacing(z=1.0, y=1.0, x=1.0),
            channels={"source": "Phase3D"},
            targets={
                "nucleus": TargetConfig(
                    gene="H2B",
                    organelle="nucleus",
                    display_name="Nucleus",
                    target_channel="Nuclei",
                    stores=StoreLocations(train=train_zarr, test=test_zarr),
                    splits="splits/nucleus.yaml",
                ),
            },
        )

    def test_sample_does_not_overwrite_the_full_archive(self, tmp_path):
        """`pack` then `sample` into one --output-root must leave both archives.

        They used to resolve to the identical dst_ozx_path — the `_sample`
        suffix lived only on the throwaway tmp zarr — so a reviewer subset
        silently replaced a multi-GB release archive, and write_pack_manifest
        then recorded the sample's sha256/bytes as the dataset's.
        """
        from dynacell.distribution import pack_dataset

        train_zarr = tmp_path / "train_cell.zarr"
        test_zarr = tmp_path / "test_cell.zarr"
        _make_fixture_zarr(train_zarr, n_pos=4, t=3)
        _make_fixture_zarr(test_zarr, n_pos=4, t=3)
        out = tmp_path / "out"

        with patch("dynacell.distribution.ozx.get_manifest", return_value=self._manifest(train_zarr, test_zarr)):
            full = pack_dataset("test-dataset", output_root=out, mode="all")
            full_bytes = {r.dst_ozx_path: r.dst_ozx_path.stat().st_size for r in full}
            sample = pack_dataset(
                "test-dataset", output_root=out, mode="sample", fov_limit=1, t_limit=1, overwrite=True
            )

        full_paths = {r.dst_ozx_path for r in full}
        sample_paths = {r.dst_ozx_path for r in sample}
        assert full_paths.isdisjoint(sample_paths)
        assert all(p.name.endswith("_sample.ozx") for p in sample_paths)
        for path, size in full_bytes.items():
            assert path.exists(), f"{path} was destroyed by the sample pack"
            assert path.stat().st_size == size, f"{path} was rewritten by the sample pack"

    def test_sample_mode_subset_writer_limits_fovs_and_t(self, tmp_path):
        """``mode='sample'`` packs at most ``fov_limit`` FOVs × ``t_limit`` frames.

        The vendored ``pack_ozx`` is 1:1 over its input, so subset
        bounds come from a temp zarr written by ``_subset_zarr``
        before packing.
        """
        from dynacell.data.manifests import (
            DatasetManifest,
            StoreLocations,
            TargetConfig,
            VoxelSpacing,
        )
        from dynacell.distribution import pack_dataset

        # Source has 3 FOVs × 4 timepoints; expect 2 × 1 after sampling.
        train_zarr = tmp_path / "train.zarr"
        test_zarr = tmp_path / "test.zarr"
        _make_fixture_zarr(train_zarr, n_pos=3, t=4)
        _make_fixture_zarr(test_zarr, n_pos=3, t=4)

        manifest = DatasetManifest(
            name="sample-test",
            version="1",
            description="Test",
            cell_type="A549",
            imaging_modality="confocal",
            spacing=VoxelSpacing(z=1.0, y=1.0, x=1.0),
            channels={"source": "Phase3D"},
            targets={
                "x": TargetConfig(
                    gene="X",
                    organelle="x",
                    display_name="X",
                    target_channel="X",
                    stores=StoreLocations(train=train_zarr, test=test_zarr),
                    splits="splits/x.yaml",
                ),
            },
        )

        with patch(
            "dynacell.distribution.ozx.get_manifest",
            return_value=manifest,
        ):
            results = pack_dataset(
                "sample-test",
                output_root=tmp_path / "out",
                mode="sample",
                fov_limit=2,
                t_limit=1,
            )

        assert len(results) == 2  # train + test
        # Verify the packed archive contains the subset shape, not the source.
        for r in results:
            extracted = tmp_path / f"extracted_{r.split}.zarr"
            with zipfile.ZipFile(r.dst_ozx_path) as zf:
                zf.extractall(extracted)
            with open_ome_zarr(extracted, mode="r", layout="hcs") as plate:
                positions = list(plate.positions())
                assert len(positions) == 2  # fov_limit
                arr = positions[0][1]["0"][:]
                assert arr.shape[0] == 1  # t_limit

    def test_sample_mode_requires_both_limits(self, tmp_path):
        """Sample mode without fov_limit or t_limit raises ValueError."""
        from dynacell.data.manifests import (
            DatasetManifest,
            StoreLocations,
            TargetConfig,
            VoxelSpacing,
        )
        from dynacell.distribution import pack_dataset

        train_zarr = tmp_path / "train.zarr"
        _make_fixture_zarr(train_zarr, n_pos=1, t=1)
        manifest = DatasetManifest(
            name="t",
            version="1",
            description="t",
            cell_type="A549",
            imaging_modality="confocal",
            spacing=VoxelSpacing(z=1.0, y=1.0, x=1.0),
            channels={"source": "Phase3D"},
            targets={
                "x": TargetConfig(
                    gene="X",
                    organelle="x",
                    display_name="X",
                    target_channel="X",
                    stores=StoreLocations(train=train_zarr, test=train_zarr),
                    splits="splits/x.yaml",
                ),
            },
        )
        with patch(
            "dynacell.distribution.ozx.get_manifest",
            return_value=manifest,
        ):
            with pytest.raises(ValueError, match="sample mode requires"):
                pack_dataset(
                    "t",
                    output_root=tmp_path / "out",
                    mode="sample",
                    fov_limit=2,
                    # t_limit deliberately omitted.
                )


class TestSync:
    """sync_to_s3 hard-fails when the AWS CLI is missing or local_root is bogus."""

    def test_raises_when_aws_cli_missing(self, tmp_path, monkeypatch):
        """No `aws` on PATH → RuntimeError, not silent fallback."""
        from dynacell.distribution.sync import sync_to_s3

        monkeypatch.setattr("shutil.which", lambda _name: None)
        with pytest.raises(RuntimeError, match="aws CLI not on PATH"):
            sync_to_s3(tmp_path, bucket="b", prefix="p")

    def test_raises_when_local_root_missing(self, tmp_path, monkeypatch):
        """A non-directory local_root → FileNotFoundError before invoking aws."""
        from dynacell.distribution.sync import sync_to_s3

        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/aws")
        with pytest.raises(FileNotFoundError):
            sync_to_s3(tmp_path / "missing", bucket="b", prefix="p")

    def test_dry_run_passes_dryrun_flag(self, tmp_path, monkeypatch):
        """Default dry_run=True invokes `aws s3 sync ... --dryrun`."""
        from dynacell.distribution import sync as sync_mod

        monkeypatch.setattr(sync_mod.shutil, "which", lambda _name: "/usr/bin/aws")
        captured = {}

        def fake_run(cmd, check):
            captured["cmd"] = cmd

            class _Result:
                returncode = 0

            return _Result()

        monkeypatch.setattr(sync_mod.subprocess, "run", fake_run)
        sync_mod.sync_to_s3(tmp_path, bucket="b", prefix="p")
        assert "--dryrun" in captured["cmd"]


class TestSplitCsv:
    """``_split_csv`` strips whitespace and drops empty tokens."""

    def test_strips_whitespace_and_empties(self):
        """Whitespace around CSV tokens is stripped; empties dropped."""
        from dynacell.distribution.cli import _split_csv

        assert _split_csv("train, test") == ["train", "test"]
        assert _split_csv(" sec61b ,, tomm20 ,") == ["sec61b", "tomm20"]

    def test_falsy_returns_none(self):
        """``None``, empty, and whitespace-only inputs collapse to ``None``."""
        from dynacell.distribution.cli import _split_csv

        assert _split_csv(None) is None
        assert _split_csv("") is None
        assert _split_csv("   ,  , ") is None


class TestVerifyPublic:
    """verify_public extracts contentUrls and surfaces head_object errors."""

    def test_iter_content_urls(self):
        """Walks distribution[*].contentUrl strings."""
        from dynacell.distribution.verify import _iter_content_urls

        jsonld = {
            "distribution": [
                {"contentUrl": "s3://b/k1"},
                {"contentUrl": "s3://b/k2"},
                {"@type": "cr:FileObject"},  # no URL → skipped
            ]
        }
        urls = _iter_content_urls(jsonld)
        assert urls == ["s3://b/k1", "s3://b/k2"]


class TestManifest:
    """MANIFEST.json shape + serialization."""

    def test_manifest_shape(self, tmp_path):
        """One entry per PackResult; sha256 + size populated."""
        from dynacell.distribution import (
            PackResult,
            write_pack_manifest,
        )

        results = [
            PackResult(
                dataset="test",
                target="sec61b",
                split="train",
                src_zarr_path=Path("/src/SEC61B.zarr"),
                dst_ozx_path=tmp_path / "out.ozx",
                bytes=1234,
                sha256="a" * 64,
                ozx_version="0.5",
            )
        ]
        manifest_path = tmp_path / "MANIFEST.json"
        write_pack_manifest("test", results, manifest_path)

        loaded = json.loads(manifest_path.read_text())
        assert loaded["dataset"] == "test"
        assert len(loaded["entries"]) == 1
        entry = loaded["entries"][0]
        assert entry["target"] == "sec61b"
        assert entry["split"] == "train"
        assert entry["sha256"] == "a" * 64
        assert entry["ozx_version"] == "0.5"

    def test_split_scoped_packs_accumulate(self, tmp_path):
        """A second --splits-scoped pack keeps the first call's archive.

        ``pack --splits train`` then ``pack --splits test`` leaves both .ozx on
        disk; a truncating write would publish a checksum ledger naming only the
        second.
        """
        from dynacell.distribution import (
            PackResult,
            write_pack_manifest,
        )

        def _result(split: str, digest: str) -> PackResult:
            ozx = tmp_path / f"{split}.ozx"
            ozx.write_bytes(b"archive")
            return PackResult(
                dataset="test",
                target="sec61b",
                split=split,
                src_zarr_path=Path(f"/src/{split}/SEC61B.zarr"),
                dst_ozx_path=ozx,
                bytes=7,
                sha256=digest * 64,
                ozx_version="0.5",
            )

        manifest_path = tmp_path / "MANIFEST.json"
        write_pack_manifest("test", [_result("train", "a")], manifest_path)
        write_pack_manifest("test", [_result("test", "b")], manifest_path)

        entries = json.loads(manifest_path.read_text())["entries"]
        assert [e["split"] for e in entries] == ["test", "train"]

    def test_a_deleted_archive_is_dropped_from_the_ledger(self, tmp_path):
        """A prior entry survives only while its .ozx is still on disk."""
        from dynacell.distribution import (
            PackResult,
            write_pack_manifest,
        )

        gone = tmp_path / "train.ozx"
        gone.write_bytes(b"archive")
        stale = PackResult(
            dataset="test",
            target="sec61b",
            split="train",
            src_zarr_path=Path("/src/SEC61B.zarr"),
            dst_ozx_path=gone,
            bytes=7,
            sha256="a" * 64,
            ozx_version="0.5",
        )
        manifest_path = tmp_path / "MANIFEST.json"
        write_pack_manifest("test", [stale], manifest_path)
        gone.unlink()

        kept = tmp_path / "test.ozx"
        kept.write_bytes(b"archive")
        write_pack_manifest(
            "test",
            [dataclasses.replace(stale, split="test", dst_ozx_path=kept, sha256="b" * 64)],
            manifest_path,
        )

        entries = json.loads(manifest_path.read_text())["entries"]
        assert [e["split"] for e in entries] == ["test"]
