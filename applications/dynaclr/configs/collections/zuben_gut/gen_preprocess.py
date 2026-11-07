"""Generate + submit `viscy preprocess` SLURM jobs for the 25 v3 gut stores.

Writes per-channel normalization stats (fov/dataset/timepoint) into each store's
``.zattrs``. One SLURM job per store (each store is a single large FOV).

Run::

    uv run --no-sync python applications/dynaclr/configs/collections/zuben_gut/gen_preprocess.py
    bash <this_dir>/preprocess/submit_all.sh
"""

import glob
from pathlib import Path

V3_ROOT = "/hpc/projects/organelle_phenotyping/datasets/zuben_gut_development"
REPO = "/hpc/mydata/eduardo.hirata/repos/viscy"
VENV = f"{REPO}/.venv-dynaclr"
HERE = Path(__file__).resolve().parent
WORKDIR = HERE / "preprocess"

JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=pp_{stem}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output={workdir}/slurm_pp_{stem}_%j.out

export PYTHONNOUSERSITE=1
export UV_PROJECT_ENVIRONMENT={venv}

uv run --project "{repo}" --package dynaclr \\
    viscy preprocess --data_path "{store}" \\
    --channel_names=-1 --num_workers 32 --block_size 32
"""


def main() -> None:
    """Emit one `viscy preprocess` SLURM job per v3 store plus a submit driver."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    stores = sorted(glob.glob(f"{V3_ROOT}/*.zarr"))
    submit = ["#!/bin/bash", "set -euo pipefail", ""]
    for store in stores:
        stem = Path(store).name.replace(".zarr", "")
        job = WORKDIR / f"pp_{stem}.sh"
        job.write_text(JOB_TEMPLATE.format(stem=stem, workdir=WORKDIR, venv=VENV, repo=REPO, store=store))
        submit.append(f"sbatch {job}")
    (WORKDIR / "submit_all.sh").write_text("\n".join(submit) + "\n")
    print(f"Generated {len(stores)} preprocess jobs under {WORKDIR}")
    print(f"Submit: bash {WORKDIR / 'submit_all.sh'}")


if __name__ == "__main__":
    main()
