"""60-second NCCL all-reduce smoke test for dynacell training preflight.

Maps Slurm's per-task env vars (``SLURM_PROCID``, ``SLURM_NTASKS``,
``SLURM_LOCALID``) to the ``env://`` init vars ``torch.distributed`` expects
(``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``), then runs a single ``all_reduce``
+ ``barrier`` with a 60-second init timeout. Exits non-zero (via unhandled
``RuntimeError``) on hang or any NCCL error — the calling sbatch script uses
that to abort before the 30-minute watchdog timeout on a bad node.
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> int:
    """Return 0 on successful NCCL init + all_reduce; raise on hang or NCCL error.

    Non-zero exit (via unhandled exception) signals the calling sbatch script
    to abort before the main training srun, skipping the 30-minute watchdog
    wait on a bad node.
    """
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))
    t = torch.ones(1, device="cuda")
    dist.all_reduce(t)
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"[nccl-smoke] OK world_size={dist.get_world_size()} sum={t.item()}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
