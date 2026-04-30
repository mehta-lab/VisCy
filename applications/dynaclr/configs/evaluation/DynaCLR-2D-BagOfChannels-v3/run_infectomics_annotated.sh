#!/bin/bash
# Wave-1 evaluation: DynaCLR-2D-BagOfChannels-v3 x infectomics-annotated.
# Sibling comparison run alongside the 2D-MIP-BagOfChannels family.
#
# Submit:
#   sbatch applications/dynaclr/configs/evaluation/DynaCLR-2D-BagOfChannels-v3/run_infectomics_annotated.sh

#SBATCH --job-name=eval_w1_2dboc_v3_infectomics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --output=/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/v3/evaluations/infectomics-annotated/nextflow_logs/%x-%j.out

export PYTHONNOUSERSITE=1

module load nextflow/24.10.5

WORKSPACE="/hpc/mydata/eduardo.hirata/repos/viscy"
CONFIG="$WORKSPACE/applications/dynaclr/configs/evaluation/DynaCLR-2D-BagOfChannels-v3/infectomics-annotated.yaml"
LOGDIR="/hpc/projects/organelle_phenotyping/models/SEC61_TOMM20_G3BP1_Sensor/time_interval/dynaclr_gfp_rfp_Ph/v3/evaluations/infectomics-annotated/nextflow_logs"

mkdir -p "$LOGDIR"
cd "$LOGDIR"

nextflow run "$WORKSPACE/applications/dynaclr/nextflow/main.nf" -entry evaluation --eval_config "$CONFIG" --workspace_dir "$WORKSPACE" -resume
