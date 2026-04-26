"""Real-DDP integration tests for ``BatchedConcatDataModule``.

Spawn two ranks via ``torch.multiprocessing`` (``gloo`` backend) so the
joint loader is exercised through a genuine
``torch.distributed.init_process_group``. The monkeypatch-based DDP
tests in ``test_combined.py`` cover the sampler-attachment contract but
not the worker-spawn × real-DDP interaction.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from iohub import open_ome_zarr

from viscy_data import HCSDataModule, ShardedDistributedSampler
from viscy_data.combined import BatchedConcatDataModule

WORLD_SIZE = 2
BATCH_SIZE = 4
CHANNEL_NAMES = ["Phase3D", "Nuclei"]


def _kill_survivors(processes, grace: float) -> None:
    """Terminate-then-kill any still-alive child processes.

    Idempotent: a no-op when every process has already exited.
    """
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
            proc.join(grace)
            if proc.is_alive():
                proc.kill()


def _build_zarr(path: Path) -> None:
    """Build a deterministic 4-FOV HCS zarr suitable for the smoke."""
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=CHANNEL_NAMES) as ds:
        rng = np.random.default_rng(42)
        for fov in ("0", "1", "2", "3"):
            pos = ds.create_position("A", "1", fov)
            img = rng.random((1, len(CHANNEL_NAMES), 8, 64, 64)).astype(np.float32)
            pos.create_image("0", img, chunks=(1, 1, 1, 64, 64))


def _make_dm(data_path: str, num_workers: int, mmap_preload: bool, scratch_dir: str | None) -> HCSDataModule:
    kwargs: dict = dict(
        data_path=data_path,
        source_channel=["Phase3D"],
        target_channel=["Nuclei"],
        z_window_size=4,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        yx_patch_size=(32, 32),
        split_ratio=0.8,
        pin_memory=False,
    )
    if mmap_preload:
        kwargs["mmap_preload"] = True
        if scratch_dir is not None:
            kwargs["scratch_dir"] = Path(scratch_dir)
    return HCSDataModule(**kwargs)


def _worker(
    rank: int,
    init_file: str,
    data_path: str,
    out_path: str,
    num_workers: int,
    mmap_preload: bool,
    scratch_dir: str | None,
) -> None:
    """One DDP rank. Writes ``OK`` / ``FAIL`` + findings to ``{out_path}.{rank}``."""
    findings: list[str] = []

    def record(msg: str) -> None:
        findings.append(f"[rank {rank}] {msg}")

    ok = True
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            world_size=WORLD_SIZE,
            rank=rank,
        )
        record(f"process group up, num_workers={num_workers}, mmap_preload={mmap_preload}")

        dm_a = _make_dm(data_path, num_workers, mmap_preload, scratch_dir)
        dm_b = _make_dm(data_path, num_workers, mmap_preload, scratch_dir)
        batched = BatchedConcatDataModule(data_modules=[dm_a, dm_b])

        # Lightning wraps ``prepare_data`` in a barrier so non-rank-0 ranks
        # don't race past rank 0's mmap writer; reproduce that pattern so
        # ``mmap_preload=True`` doesn't trip ``_check_mmap_cache_ready``
        # before ``.done`` exists.
        if rank == 0:
            batched.prepare_data()
        dist.barrier()
        batched.setup(stage="fit")

        train_loader = batched.train_dataloader()
        val_loader = batched.val_dataloader()
        assert isinstance(train_loader.sampler, ShardedDistributedSampler)
        assert isinstance(val_loader.sampler, ShardedDistributedSampler)
        assert train_loader.sampler.shuffle is True
        assert val_loader.sampler.shuffle is False
        assert train_loader.sampler.rank == rank
        assert train_loader.sampler.num_replicas == WORLD_SIZE

        n_batches = 0
        for batch in train_loader:
            assert isinstance(batch, list)
            for mb in batch:
                assert "_dataset_idx" in mb
                assert "source" in mb
                assert mb["source"].ndim == 5  # (B, C, Z, Y, X)
            assert sum(mb["source"].shape[0] for mb in batch) == BATCH_SIZE
            n_batches += 1
            if n_batches >= 3:
                break
        record(f"iterated {n_batches} train batches")

        # Cross-rank disjointness via all_gather_object on the sampler indices.
        train_loader.sampler.set_epoch(0)
        local_indices = list(iter(train_loader.sampler))
        all_indices: list[list[int] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(all_indices, local_indices)
        if rank == 0:
            seen: set[int] = set()
            for r, ranks_indices in enumerate(all_indices):
                assert ranks_indices is not None
                for idx in ranks_indices:
                    assert idx not in seen, f"rank {r} reused index {idx}"
                    seen.add(idx)
            record(f"cross-rank disjoint, union={len(seen)}")

        val_loader.sampler.set_epoch(0)
        n_val = 0
        for _ in val_loader:
            n_val += 1
            if n_val >= 2:
                break
        record(f"iterated {n_val} val batches")
    except Exception:
        ok = False
        findings.append(f"[rank {rank}] FAILED\n{traceback.format_exc()}")
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        Path(f"{out_path}.{rank}").write_text(("OK\n" if ok else "FAIL\n") + "\n".join(findings) + "\n")


@pytest.mark.parametrize(
    "num_workers,mmap_preload",
    [
        pytest.param(0, False, id="nw0_no_mmap"),
        pytest.param(2, False, id="nw2_no_mmap"),
        pytest.param(2, True, id="nw2_mmap_preload"),
    ],
)
def test_batched_concat_real_ddp_iter_does_not_hang(
    tmp_path_factory: pytest.TempPathFactory,
    num_workers: int,
    mmap_preload: bool,
) -> None:
    """Spawn 2 ranks, iterate the joint loader, assert no deadlock.

    Locks down the joint loader's collate, sampler-attachment, and
    rank-0 ``prepare_data`` ordering under real DDP + multi-worker +
    ``mmap_preload``. The GPU/NCCL-specific deadlock that PR #413
    fixed (pin-memory thread × thread-shim worker context under CUDA)
    is not reproducible on CPU/gloo and needs a GPU runner to catch a
    revert of ``use_thread_workers=True`` directly.
    """
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork start_method not available (Windows)")

    work_dir = tmp_path_factory.mktemp(f"ddp_{num_workers}_{int(mmap_preload)}")
    data_path = work_dir / "smoke.zarr"
    _build_zarr(data_path)
    init_file = work_dir / "pg_init"
    out_base = work_dir / "result"
    scratch_dir = work_dir / "scratch" if mmap_preload else None
    if scratch_dir is not None:
        scratch_dir.mkdir()

    # ``start_method="fork"`` because pytest imports tests under
    # ``--import-mode=importlib``, whose path can't be re-resolved in a
    # spawn child (``ModuleNotFoundError: 'packages'``). ``mp.spawn``
    # only supports ``spawn``, so go through ``mp.start_processes``.
    ctx = mp.start_processes(
        _worker,
        args=(
            str(init_file),
            str(data_path),
            str(out_base),
            num_workers,
            mmap_preload,
            str(scratch_dir) if scratch_dir is not None else None,
        ),
        nprocs=WORLD_SIZE,
        join=False,
        daemon=False,
        start_method="fork",
    )

    # ``ctx.join`` returns ``False`` as soon as any rank exits, so a
    # single ``ctx.join(timeout=120)`` would terminate prematurely on
    # the first rank's clean exit; loop on a wall-time deadline.
    deadline = time.monotonic() + 120
    try:
        while not ctx.join(timeout=max(0.05, deadline - time.monotonic()), grace_period=5):
            if time.monotonic() >= deadline:
                _kill_survivors(ctx.processes, grace=2)
                pytest.fail("DDP test hung past 120s; killed surviving ranks")
    finally:
        _kill_survivors(ctx.processes, grace=1)

    failures: list[str] = []
    for rank in range(WORLD_SIZE):
        text = Path(f"{out_base}.{rank}").read_text()
        if not text.startswith("OK"):
            failures.append(f"--- rank {rank} ---\n{text}")
    if failures:
        pytest.fail("\n\n".join(failures))
