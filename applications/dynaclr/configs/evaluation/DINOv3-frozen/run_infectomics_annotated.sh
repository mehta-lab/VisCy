#!/bin/bash
# Wave-1 evaluation: DINOv3-frozen × infectomics-annotated.
#
# Baseline that pulls raw DINOv3 convnext-tiny weights from HuggingFace —
# no contrastive fine-tuning, no projection head. Tests whether DynaCLR's
# training adds value beyond pre-trained DINOv3 features.
#
# Submit:
#   sbatch applications/dynaclr/configs/evaluation/DINOv3-frozen/run_infectomics_annotated.sh

#SBATCH --job-name=eval_w1_dinov3_frozen_infectomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --output=/hpc/projects/organelle_phenotyping/models/DINOv3-frozen/evaluations/infectomics-annotated/nextflow_logs/%x-%j.out

export PYTHONNOUSERSITE=1

module load nextflow/24.10.5

WORKSPACE="/hpc/mydata/eduardo.hirata/repos/viscy"
CONFIG="$WORKSPACE/applications/dynaclr/configs/evaluation/DINOv3-frozen/infectomics-annotated.yaml"
LOGDIR="/hpc/projects/organelle_phenotyping/models/DINOv3-frozen/evaluations/infectomics-annotated/nextflow_logs"

mkdir -p "$LOGDIR"
cd "$LOGDIR"

nextflow run "$WORKSPACE/applications/dynaclr/nextflow/main.nf" -entry evaluation --eval_config "$CONFIG" --workspace_dir "$WORKSPACE" -resume
