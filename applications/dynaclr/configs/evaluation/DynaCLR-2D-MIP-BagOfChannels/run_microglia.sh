#!/bin/bash
# Wave-2 evaluation: DynaCLR-2D-MIP-BagOfChannels x microglia.
# Applies LC pipelines from the central registry's infectomics sub-registry:
#   /hpc/projects/.../linear_classifiers/DynaCLR-2D-MIP-BagOfChannels/infectomics/latest
# No annotations to append. Marker mismatch (microglia has Brightfield,
# Phase3D, Retardance vs registry trained on G3BP1/SEC61B/Phase3D/viral_sensor)
# means only Phase3D cells get predictions. Coverage report logged.
#
# Submit:
#   sbatch applications/dynaclr/configs/evaluation/DynaCLR-2D-MIP-BagOfChannels/run_microglia.sh

#SBATCH --job-name=eval_w2_2dmip_microglia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out

export PYTHONNOUSERSITE=1

module load nextflow/24.10.5

WORKSPACE="/hpc/mydata/eduardo.hirata/repos/viscy"
CONFIG="$WORKSPACE/applications/dynaclr/configs/evaluation/DynaCLR-2D-MIP-BagOfChannels/microglia.yaml"
LOGDIR="/hpc/projects/organelle_phenotyping/models/DynaCLR-2D-MIP-BagOfChannels/evaluations/microglia/nextflow_logs"

mkdir -p "$LOGDIR"
cd "$LOGDIR"

nextflow run "$WORKSPACE/applications/dynaclr/nextflow/main.nf" -entry evaluation \
    --eval_config "$CONFIG" \
    --workspace_dir "$WORKSPACE" \
    -resume
