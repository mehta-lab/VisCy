#!/bin/bash
# Legacy transitional config; new benchmark launches should use Dynacell.
# See: applications/dynacell/examples/configs/sec61b/

#SBATCH --job-name=VSCyto3D_SEC61B
#SBATCH --time=20-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --constraint=h200
#SBATCH --output=/hpc/projects/comp.micro/virtual_staining/models/dynacell_cytoland/ipsc/sec61b/vscyto3d/slurm/%j.out
#SBATCH --error=/hpc/projects/comp.micro/virtual_staining/models/dynacell_cytoland/ipsc/sec61b/vscyto3d/slurm/%j.err

mkdir -p -m 775 /hpc/projects/comp.micro/virtual_staining/models/dynacell_cytoland/ipsc/sec61b/vscyto3d/slurm
mkdir -p -m 775 /hpc/projects/comp.micro/virtual_staining/models/dynacell_cytoland/ipsc/sec61b/vscyto3d/checkpoints

ml uv

export PYTHONUNBUFFERED=1

nvidia-smi
uv run python -m cytoland fit --config applications/cytoland/examples/configs/dynacell/fit_vscyto3d_sec61b.yml
