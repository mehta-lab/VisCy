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

EMBEDDINGS_DIR=/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_01_28_A549_G3BP1_ZIKV_DENV/4-phenotyping/predictions/DynaCLR-2D-BagOfChannels-timeaware/v3
CSV=/hpc/projects/organelle_phenotyping/datasets/annotations/2025_01_28_A549_G3BP1_ZIKV_DENV/2025_01_28_A549_G3BP1_ZIKV_DENV_combined_annotations.csv
PREFIX=annotated_
MERGE_KEY=fov_name
# -----------------------------------------------------------------------------

scontrol show job $SLURM_JOB_ID

echo "Embeddings: $EMBEDDINGS"
echo "CSV: $CSV"
echo "Prefix: $PREFIX"
echo "Merge key: $MERGE_KEY"
echo ""

#For each zarr in the embeddings path, append the obs to the csv
for ZARR in $EMBEDDINGS_DIR/*.zarr; do
    echo "Appending obs to $ZARR"
    uv run --project "$WORKSPACE_DIR" --package dynaclr --extra eval \
        dynaclr append-obs \
        -e "$ZARR" \
        --csv "$CSV" \
        --prefix "$PREFIX" \
        --merge-key "fov_name" \
        --merge-key "track_id" \
        --merge-key "id"

done
