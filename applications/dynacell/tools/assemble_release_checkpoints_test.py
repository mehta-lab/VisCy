"""Tests for the public checkpoint-zoo assembler.

This tool had no test at all, while carrying findings about what it writes and
which checkpoint it resolves. The cases here cover the contract that matters
most operationally: ``--dest`` is a public share, so a dry run must leave it
untouched, and a run that resolves nothing must refuse to publish a manifest
rather than replace a good one with dead rows.
"""

from __future__ import annotations

import csv

import pytest
from assemble_release_checkpoints import main, write_manifest

_PLAN = [
    {
        "train_pub": "joint",
        "organelle": "mito",
        "model_slug": "celldiff",
        "arch": "CELL-Diff",
        "provenance": "raw",
        "status": "resolved",
        "pub_name": "epoch=19-step=192800.ckpt",
        "size_gb": 1.119,
        "dst_ckpt": "/dest/models/joint/mito/celldiff/epoch=19-step=192800.ckpt",
        "copy_src": "/src/checkpoints/epoch=19-step=192800.ckpt",
        "config_src": "/src/config.yaml",
    }
]


def _run(monkeypatch, argv: list[str]) -> None:
    """Drive the real console entry point; main() reads sys.argv."""
    monkeypatch.setattr("sys.argv", ["assemble_release_checkpoints.py", *argv])
    main()


def test_dry_run_leaves_the_destination_untouched(tmp_path, capsys, monkeypatch):
    """A dry run must not create --dest or write a CSV into it.

    --dest points at the public release share. Materializing a directory and a
    checkpoints_manifest_dryrun.csv there is a write, however small, from a mode
    whose whole contract is that it does not write.
    """
    dest = tmp_path / "release"
    _run(monkeypatch, ["--dest", str(dest)])

    assert not dest.exists(), sorted(p.name for p in dest.rglob("*"))
    out = capsys.readouterr().out
    assert "manifest preview: stdout" in out
    # The preview still reaches the operator, just on stdout.
    assert "train_set" in out


def test_manifest_flag_still_writes_a_file(tmp_path, monkeypatch):
    """--manifest PATH is the explicit opt-in that captures the preview."""
    dest = tmp_path / "release"
    manifest = tmp_path / "preview.csv"
    _run(monkeypatch, ["--dest", str(dest), "--manifest", str(manifest)])

    assert manifest.is_file()
    assert not dest.exists()
    rows = list(csv.reader(manifest.read_text().splitlines()))
    assert rows[0][0] == "train_set"
    assert len(rows) > 1


def test_write_manifest_to_stdout_creates_no_directory(tmp_path, capsys):
    """write_manifest(plan, None) prints and touches nothing on disk."""
    write_manifest(_PLAN, None)
    assert sorted(tmp_path.iterdir()) == []
    assert "celldiff" in capsys.readouterr().out


def test_unknown_model_slug_is_rejected(tmp_path, monkeypatch):
    """A bad --models filter must fail loudly, not resolve zero and continue."""
    with pytest.raises(SystemExit):
        _run(monkeypatch, ["--dest", str(tmp_path / "release"), "--models", "not-a-model"])
