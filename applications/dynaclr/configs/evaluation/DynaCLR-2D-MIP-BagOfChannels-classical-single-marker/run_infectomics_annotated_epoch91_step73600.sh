#!/bin/bash
# Wave-1 evaluation: DynaCLR-2D-MIP-BagOfChannels classical-single-marker x infectomics-annotated.
# Checkpoint: epoch=91-step=73600 (run id dt0ln63r).
#
# Submit:
#   sbatch applications/dynaclr/configs/evaluation/DynaCLR-2D-MIP-BagOfChannels-classical-single-marker/run_infectomics_annotated_epoch91_step73600.sh

#SBATCH --job-name=eval_w1_2dmip_classical_sm_infectomics_epoch91_step73600
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
CONFIG="$WORKSPACE/applications/dynaclr/configs/evaluation/DynaCLR-2D-MIP-BagOfChannels-classical-single-marker/infectomics-annotated_epoch91_step73600.yaml"
LOGDIR="/hpc/projects/organelle_phenotyping/models/DynaCLR-2D-MIP-BagOfChannels/2d-mip-classical-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker/evaluations/infectomics-annotated_epoch91_step73600/nextflow_logs"

mkdir -p "$LOGDIR"
cd "$LOGDIR"

nextflow run "$WORKSPACE/applications/dynaclr/nextflow/main.nf" -entry evaluation \
    --eval_config "$CONFIG" \
    --workspace_dir "$WORKSPACE" \
    -resume
