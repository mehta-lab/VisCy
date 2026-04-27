#!/bin/bash
# Wave-1 evaluation: DynaCLR-2D-MIP-BagOfChannels x infectomics-annotated.
# Trains linear classifiers on the 14 ZIKV+DENV infectomics experiments and
# publishes them to the central LC registry at
#   /hpc/projects/organelle_phenotyping/models/linear_classifiers/DynaCLR-2D-MIP-BagOfChannels/vN/
# with a `latest` symlink updated atomically at the end of the LC step.
#
# Submit:
#   sbatch applications/dynaclr/configs/evaluation/DynaCLR-2D-MIP-BagOfChannels/run_infectomics_annotated.sh

#SBATCH --job-name=eval_w1_2dmip_infectomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
# Wrapper hosts the Nextflow head + any local-executor processes (per-experiment
# PLOT runs locally; PLOT_COMBINED + REDUCE_COMBINED + LC + PREDICT go to slurm).
# 32G + 4 cpus is enough for 19 sequential per-experiment plots on 350k cells.

export PYTHONNOUSERSITE=1

module load nextflow/24.10.5

WORKSPACE="/hpc/mydata/eduardo.hirata/repos/viscy"
CONFIG="$WORKSPACE/applications/dynaclr/configs/evaluation/DynaCLR-2D-MIP-BagOfChannels/infectomics-annotated.yaml"
LOGDIR="/hpc/projects/organelle_phenotyping/models/DynaCLR-2D-MIP-BagOfChannels/2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11/evaluations/infectomics-annotated/nextflow_logs"

mkdir -p "$LOGDIR"
cd "$LOGDIR"

nextflow run "$WORKSPACE/applications/dynaclr/nextflow/main.nf" -entry evaluation \
    --eval_config "$CONFIG" \
    --workspace_dir "$WORKSPACE" \
    -resume
