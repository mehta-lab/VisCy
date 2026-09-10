"""Tests for the checkpoint content hash shared by prediction writers and caches."""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import pytest

from viscy_utils import prediction_metadata
from viscy_utils.prediction_metadata import checkpoint_sha256_12


def test_checkpoint_sha256_12(tmp_path: Path) -> None:
    """Returns the first 12 hex chars of sha256; differs for different content."""
    file_a = tmp_path / "a.ckpt"
    file_b = tmp_path / "b.ckpt"
    file_a.write_bytes(b"model-weights-a")
    file_b.write_bytes(b"model-weights-b")

    h_a = checkpoint_sha256_12(file_a)
    h_b = checkpoint_sha256_12(file_b)
    assert len(h_a) == 12
    assert len(h_b) == 12
    assert h_a != h_b
    assert checkpoint_sha256_12(file_a) == h_a  # deterministic


def test_checkpoint_sha256_12_writes_and_reuses_sidecar(tmp_path: Path, monkeypatch) -> None:
    """First call writes ``<ckpt>.sha256`` with the file's size and mtime; second call skips the hash."""
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    h1 = checkpoint_sha256_12(ckpt)
    sidecar = tmp_path / "last.ckpt.sha256"
    recorded = json.loads(sidecar.read_text())
    assert recorded["sha256"][:12] == h1
    assert len(recorded["sha256"]) == 64
    stat = ckpt.stat()
    assert recorded["size"] == stat.st_size
    assert recorded["mtime_ns"] == stat.st_mtime_ns
    assert recorded["ctime_ns"] == stat.st_ctime_ns
    assert recorded["ino"] == stat.st_ino

    calls = {"n": 0}
    real_sha256 = hashlib.sha256

    def tracking_sha256(*args, **kwargs):
        calls["n"] += 1
        return real_sha256(*args, **kwargs)

    monkeypatch.setattr(prediction_metadata.hashlib, "sha256", tracking_sha256)
    h2 = checkpoint_sha256_12(ckpt)
    assert h2 == h1
    assert calls["n"] == 0


def test_checkpoint_sha256_12_recomputes_when_sidecar_older(tmp_path: Path) -> None:
    """Newer ckpt mtime invalidates the sidecar and forces a recompute."""
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights-v1")
    h1 = checkpoint_sha256_12(ckpt)

    ckpt.write_bytes(b"weights-v2")
    sidecar = tmp_path / "last.ckpt.sha256"
    old = sidecar.stat().st_mtime
    os.utime(ckpt, (old + 10, old + 10))

    h2 = checkpoint_sha256_12(ckpt)
    assert h2 != h1
    assert json.loads(sidecar.read_text())["sha256"][:12] == h2


def test_checkpoint_sha256_12_recomputes_after_replacement_with_older_mtime(tmp_path: Path) -> None:
    """A same-size checkpoint copied in with a preserved, older mtime must not reuse the old digest.

    Only exact size and ``st_mtime_ns`` equality qualifies the sidecar; being
    newer than the checkpoint is not evidence the content is unchanged.
    """
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights-v1")
    h1 = checkpoint_sha256_12(ckpt)
    sidecar = tmp_path / "last.ckpt.sha256"

    ckpt.write_bytes(b"weights-v2")  # same size as v1
    older = sidecar.stat().st_mtime - 3600
    os.utime(ckpt, (older, older))

    h2 = checkpoint_sha256_12(ckpt)
    assert h2 == hashlib.sha256(b"weights-v2").hexdigest()[:12]
    assert h2 != h1


def test_checkpoint_sha256_12_recomputes_after_replacement_preserving_size_and_mtime(tmp_path: Path) -> None:
    """A same-size copy that carries the old mtime along still gets a new ctime, so the sidecar is dropped."""
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights-v1")
    h1 = checkpoint_sha256_12(ckpt)
    before = ckpt.stat()
    replacement = tmp_path / "replacement.ckpt"
    replacement.write_bytes(b"weights-v2")
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    time.sleep(0.05)  # ctime advances per kernel tick, not per write

    shutil.copy2(replacement, ckpt)

    assert ckpt.stat().st_mtime_ns == before.st_mtime_ns
    assert ckpt.stat().st_size == before.st_size
    h2 = checkpoint_sha256_12(ckpt)
    assert h2 == hashlib.sha256(b"weights-v2").hexdigest()[:12]
    assert h2 != h1


@pytest.mark.parametrize("content", ["not-a-hex-digest\n", "a" * 64 + "\n", '{"sha256": "abc"}\n'])
def test_checkpoint_sha256_12_ignores_unusable_sidecars(tmp_path: Path, content: str) -> None:
    """Corrupt, legacy plain-hex, and incomplete sidecars are treated as missing and recomputed."""
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    sidecar = tmp_path / "last.ckpt.sha256"
    sidecar.write_text(content)

    h = checkpoint_sha256_12(ckpt)
    assert h == hashlib.sha256(b"weights").hexdigest()[:12]
    assert json.loads(sidecar.read_text())["sha256"][:12] == h


def test_checkpoint_sha256_12_can_leave_the_sidecar_alone(tmp_path: Path) -> None:
    """A read-only preview gets the digest without leaving a sidecar behind."""
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"weights")
    sidecar = tmp_path / "model.ckpt.sha256"

    assert checkpoint_sha256_12(ckpt, write_sidecar=False) == hashlib.sha256(b"weights").hexdigest()[:12]
    assert not sidecar.exists()
    assert checkpoint_sha256_12(ckpt) == hashlib.sha256(b"weights").hexdigest()[:12]
    assert sidecar.exists()


def test_checkpoint_sha256_12_read_only_dir(tmp_path: Path) -> None:
    """Read-only parent dir does not raise; digest still returned."""
    ckpt_dir = tmp_path / "frozen"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "last.ckpt"
    ckpt.write_bytes(b"weights")
    os.chmod(ckpt_dir, 0o555)
    try:
        h = checkpoint_sha256_12(ckpt)
        assert len(h) == 12
        assert not (ckpt_dir / "last.ckpt.sha256").exists()
    finally:
        os.chmod(ckpt_dir, 0o755)
