#!/bin/bash
# SLURM array job: generate DINOv3 + DynaCLR embeddings for all 5 cellanome datasets.
# Array index: 0-9 (5 datasets × 2 models)
#   0-4  → DINOv3
#   5-9  → DynaCLR
#
# Usage:
#   sbatch embed_all.sh
#   # or a single task interactively:
#   SLURM_ARRAY_TASK_ID=0 bash embed_all.sh

#SBATCH --job-name=cellanome_embed
#SBATCH --array=0-9
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/hpc/mydata/eduardo.hirata/logs/cellanome_embed_%A_%a.out
#SBATCH --error=/hpc/mydata/eduardo.hirata/logs/cellanome_embed_%A_%a.err

export PYTHONNOUSERSITE=1

REPO=/home/eduardo.hirata/repos/viscy
CFG_ROOT="${REPO}/applications/dynaclr/configs/cellanome"

DATASETS=(
    "20251203141914_P-05_R000414_FC_BH_120325_try4_Adherent_with_SRA_training_4lanes"
    "20260211112411_P-05_R000439_FC_2026_02_11_manual_loading_mixed_GFP+RFP"
    "20260220144306_P-05_R000476_FC_2026_02_20_A549_GFP_RFP_Org_Cells"
    "20260310112219_P-05_R000486_FC_2026_03_10_A549_pAL27+ISG15_off_on_DENV"
    "20260324133209_P-05_R000497_FC_2026_03_24_A549_SEC61B_G3BP1_pAL40_DENV_rerun"
)

TASK=${SLURM_ARRAY_TASK_ID}
N=${#DATASETS[@]}      # 5

DATASET_IDX=$(( TASK % N ))
MODEL_IDX=$(( TASK / N ))   # 0 = DINOv3, 1 = DynaCLR

DATASET="${DATASETS[$DATASET_IDX]}"

if [ "$MODEL_IDX" -eq 0 ]; then
    SCRIPT="${REPO}/applications/dynaclr/scripts/cellanome/embed_dinov3.py"
    CONFIG="${CFG_ROOT}/${DATASET}/embed_dinov3.yml"
else
    SCRIPT="${REPO}/applications/dynaclr/scripts/cellanome/embed_dynaclr.py"
    CONFIG="${CFG_ROOT}/${DATASET}/embed_dynaclr.yml"
fi

echo "Task ${TASK}: dataset=${DATASET} model_idx=${MODEL_IDX}"
echo "Config: ${CONFIG}"

cd "${REPO}"
uv run python "${SCRIPT}" "${CONFIG}"
