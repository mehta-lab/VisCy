#!/bin/bash
# Wave-1 evaluation: CELL-DINO-temporal-MLP-2D-BagOfChannels-v1 x infectomics-annotated.
# Checkpoint: epoch=49-step=40000 (run id 0ne38zcc).
#
# Submit:
#   sbatch applications/dynaclr/configs/evaluation/CELL-DINO-temporal-MLP-2D-BagOfChannels-v1/run_infectomics_annotated_epoch49_step40000.sh

#SBATCH --job-name=eval_w1_celldino_mlp_v1_infectomics_epoch49_step40000
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
CONFIG="$WORKSPACE/applications/dynaclr/configs/evaluation/CELL-DINO-temporal-MLP-2D-BagOfChannels-v1/infectomics-annotated_epoch49_step40000.yaml"
LOGDIR="/hpc/projects/organelle_phenotyping/models/CELL-DINO-temporal-MLP-2D-BagOfChannels-v1/cell-dino-mlp-2d-mip-ntxent-t0p5-lr1e4-bs512/evaluations/infectomics-annotated_epoch49_step40000/nextflow_logs"

mkdir -p "$LOGDIR"
cd "$LOGDIR"

nextflow run "$WORKSPACE/applications/dynaclr/nextflow/main.nf" -entry evaluation \
    --eval_config "$CONFIG" \
    --workspace_dir "$WORKSPACE" \
    -resume
