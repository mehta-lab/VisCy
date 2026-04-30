#!/bin/bash
# Wave-1: DynaCLR-2D-MIP-BagOfChannels-mixed-markers-fix-shuffler × infectomics-annotated.
# Sibling comparison run.
#
# Submit:
#   sbatch applications/dynaclr/configs/evaluation/DynaCLR-2D-MIP-BagOfChannels-mixed-markers-fix-shuffler/run_infectomics_annotated.sh

#SBATCH --job-name=eval_w1_2dmip_mmfs_infectomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --output=/hpc/projects/organelle_phenotyping/models/DynaCLR-2D-MIP-BagOfChannels/2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-mixed-markers-fix-shuffler/evaluations/infectomics-annotated/nextflow_logs/%x-%j.out

export PYTHONNOUSERSITE=1

module load nextflow/24.10.5

WORKSPACE="/hpc/mydata/eduardo.hirata/repos/viscy"
CONFIG="$WORKSPACE/applications/dynaclr/configs/evaluation/DynaCLR-2D-MIP-BagOfChannels-mixed-markers-fix-shuffler/infectomics-annotated.yaml"
LOGDIR="/hpc/projects/organelle_phenotyping/models/DynaCLR-2D-MIP-BagOfChannels/2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-mixed-markers-fix-shuffler/evaluations/infectomics-annotated/nextflow_logs"

mkdir -p "$LOGDIR"
cd "$LOGDIR"

nextflow run "$WORKSPACE/applications/dynaclr/nextflow/main.nf" -entry evaluation \
    --eval_config "$CONFIG" \
    --workspace_dir "$WORKSPACE" \
    -resume
