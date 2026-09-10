"""Tests for the checkpoint content hash shared by prediction writers and caches."""

import hashlib
import os
from pathlib import Path

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
    """First call writes ``<ckpt>.sha256``; second call skips the hash."""
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    h1 = checkpoint_sha256_12(ckpt)
    sidecar = tmp_path / "last.ckpt.sha256"
    assert sidecar.exists()
    written = sidecar.read_text().strip()
    assert written[:12] == h1
    assert len(written) == 64

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
    assert sidecar.read_text().strip()[:12] == h2


def test_checkpoint_sha256_12_ignores_corrupt_sidecar(tmp_path: Path) -> None:
    """Non-hex sidecar is treated as missing and recomputed."""
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    sidecar = tmp_path / "last.ckpt.sha256"
    sidecar.write_text("not-a-hex-digest\n")
    # Match ckpt mtime so the mtime check passes and we exercise the hex guard.
    st = ckpt.stat()
    os.utime(sidecar, (st.st_mtime, st.st_mtime))

    h = checkpoint_sha256_12(ckpt)
    assert all(c in "0123456789abcdef" for c in h)
    assert len(h) == 12
    assert sidecar.read_text().strip()[:12] == h


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
