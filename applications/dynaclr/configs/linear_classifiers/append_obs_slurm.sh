#!/bin/bash

#SBATCH --job-name=dynaclr_append_obs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-00:30:00
#SBATCH --output=slurm_%j.out

export PYTHONNOUSERSITE=1

# --- Edit these paths --------------------------------------------------------
WORKSPACE_DIR=/hpc/mydata/eduardo.hirata/repos/viscy

EMBEDDINGS=/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/DINOv3/convnext-tiny-lvd1689m/dinov3-convnext-tiny-phase_2d_embeddings.zarr
CSV=/hpc/projects/organelle_phenotyping/datasets/annotations/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV_combined_annotations.csv
PREFIX=annotated_
MERGE_KEY=id
# -----------------------------------------------------------------------------

scontrol show job $SLURM_JOB_ID

echo "Embeddings: $EMBEDDINGS"
echo "CSV: $CSV"
echo "Prefix: $PREFIX"
echo "Merge key: $MERGE_KEY"
echo ""

uv run --project "$WORKSPACE_DIR" --package dynaclr --extra eval \
    dynaclr append-obs \
    -e "$EMBEDDINGS" \
    --csv "$CSV" \
    --prefix "$PREFIX" \
    --merge-key "$MERGE_KEY"
