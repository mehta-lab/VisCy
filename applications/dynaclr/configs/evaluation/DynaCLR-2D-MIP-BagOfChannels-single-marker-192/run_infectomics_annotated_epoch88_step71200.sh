#!/bin/bash
# Wave-1 evaluation: DynaCLR-2D-MIP-BagOfChannels (single-marker, 384->192, zext16) x infectomics-annotated.
# Variant: checkpoint epoch=88-step=71200 (2026-05-08), A/B against the cached infectomics-annotated/ run.
#
# Submit:
#   sbatch applications/dynaclr/configs/evaluation/DynaCLR-2D-MIP-BagOfChannels-single-marker-192/run_infectomics_annotated_epoch88_step71200.sh

#SBATCH --job-name=eval_w1_2dmip_sm192_infectomics_epoch88_step71200
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
CONFIG="$WORKSPACE/applications/dynaclr/configs/evaluation/DynaCLR-2D-MIP-BagOfChannels-single-marker-192/infectomics-annotated_epoch88_step71200.yaml"
LOGDIR="/hpc/projects/organelle_phenotyping/models/DynaCLR-2D-MIP-BagOfChannels/2d-mip-ntxent-t0p2-lr2e5-bs256-384to192-zext16-single-marker-fix-shuffler/evaluations/infectomics-annotated_2026-05-08/nextflow_logs"

mkdir -p "$LOGDIR"
cd "$LOGDIR"

nextflow run "$WORKSPACE/applications/dynaclr/nextflow/main.nf" -entry evaluation \
    --eval_config "$CONFIG" \
    --workspace_dir "$WORKSPACE" \
    -resume
