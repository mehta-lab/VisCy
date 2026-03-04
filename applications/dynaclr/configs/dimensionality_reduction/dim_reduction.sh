#!/bin/bash

#SBATCH --job-name=dynaclr_dim_red
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm_%j.out

export PYTHONNOUSERSITE=1

# --- Edit these paths --------------------------------------------------------
WORKSPACE_DIR=/hpc/mydata/eduardo.hirata/repos/viscy
PREDICTIONS_DIR=/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_01_24_A549_G3BP1_DENV/4-phenotyping/predictions/DynaCLR-2D-BagOfChannels-timeaware/v3
# -----------------------------------------------------------------------------

scontrol show job $SLURM_JOB_ID

ZARR_FILES=(
    "$PREDICTIONS_DIR/organelle_embeddings.zarr"
    "$PREDICTIONS_DIR/phase_embeddings.zarr"
    "$PREDICTIONS_DIR/sensor_embeddings.zarr"
)

for ZARR_PATH in "${ZARR_FILES[@]}"; do
    if [ ! -d "$ZARR_PATH" ]; then
        echo "WARNING: $ZARR_PATH not found, skipping"
        continue
    fi

    echo "============================================================"
    echo "Processing: $(basename "$ZARR_PATH")"
    echo "============================================================"

    CONFIG_FILE=$(mktemp /tmp/dim_reduction_XXXXXX.yaml)
    cat > "$CONFIG_FILE" <<EOF
input_path: $ZARR_PATH
overwrite_keys: true

pca:
  n_components: 32
  normalize_features: true

phate:
  n_components: 2
  knn: 5
  decay: 40
  scale_embeddings: false
  random_state: 42
EOF

    echo "Config:"
    cat "$CONFIG_FILE"
    echo ""

    uv run --project "$WORKSPACE_DIR" --package dynaclr --extra eval \
        dynaclr reduce-dimensionality -c "$CONFIG_FILE"

    rm -f "$CONFIG_FILE"
    echo ""
done

echo "All reductions complete."
