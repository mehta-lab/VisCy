#!/bin/bash
# Predict link of the model matrix (dynaclr run-matrix chains this after train).
# Runs `dynaclr predict-triplet` for one model over its collection via
# `dynaclr predict-batch`.
#
# Positional args (passed by `dynaclr run-matrix`):
#   $1 collection      collection YAML (predict-triplet input)
#   $2 model_family    PROJECT / embedding-tree key
#   $3 run             RUN_NAME / embedding-tree key
#   $4 ckpt_name       checkpoint label (last | epochN-stepM)
#   $5 datasets_root   base under which datasets live
#   $6 checkpoint      (optional) explicit checkpoint path; empty = derive from run dir
#   $7 markers         (optional) comma-separated marker subset
#   $8 z_start         (optional) first --z-range value
#   $9 z_end           (optional) second --z-range value
#   $10 z_reduction    (optional) mip | center
#   $11 reference_pixel_size (optional) microns per pixel
#   $12 batch_size     (optional) prediction batch size
#   $13 model_type     (optional) dynaclr | foundation; empty = dynaclr
#   $14 training_config (optional) LightningCLI config for --model-type foundation
#   $15 z_window        (optional) focus-centered Z window WIDTH (mutually excl. with z_range)
#   $16 focus_channel   (optional) channel whose focus_slice centers --z-window
#   $17 z_focus_offset  (optional) fraction of --z-window below focus (default 0.3)
#   $18 reference_pixel_size_z_um (optional) model Z sampling in microns per slice
#
#SBATCH --job-name=dynaclr_predict
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00

set -euo pipefail
export PYTHONNOUSERSITE=1

WORKSPACE_DIR="${WORKSPACE_DIR:-/hpc/mydata/eduardo.hirata/repos/viscy}"
cd "$WORKSPACE_DIR"

COLLECTION="$1"
MODEL_FAMILY="$2"
RUN="$3"
CKPT_NAME="$4"
DATASETS_ROOT="$5"
CHECKPOINT="${6:-}"
MARKERS="${7:-}"
Z_START="${8:-}"
Z_END="${9:-}"
Z_REDUCTION="${10:-}"
REFERENCE_PIXEL_SIZE="${11:-}"
BATCH_SIZE="${12:-}"
MODEL_TYPE="${13:-}"
TRAINING_CONFIG="${14:-}"
Z_WINDOW="${15:-}"
FOCUS_CHANNEL="${16:-}"
Z_FOCUS_OFFSET="${17:-}"
REFERENCE_PIXEL_SIZE_Z_UM="${18:-}"

CKPT_FLAG=()
if [ -n "$CHECKPOINT" ]; then
  CKPT_FLAG=(--checkpoint "$CHECKPOINT")
fi

MODEL_TYPE_FLAG=()
if [ -n "$MODEL_TYPE" ]; then
  MODEL_TYPE_FLAG=(--model-type "$MODEL_TYPE")
fi

TRAINING_CONFIG_FLAG=()
if [ -n "$TRAINING_CONFIG" ]; then
  TRAINING_CONFIG_FLAG=(--training-config "$TRAINING_CONFIG")
fi

MARKER_FLAG=()
if [ -n "$MARKERS" ]; then
  IFS=',' read -ra _M <<< "$MARKERS"
  for _m in "${_M[@]}"; do
    MARKER_FLAG+=(--markers "$_m")
  done
fi

PREDICT_FLAG=()
if [ -n "$Z_START" ] || [ -n "$Z_END" ]; then
  if [ -z "$Z_START" ] || [ -z "$Z_END" ]; then
    echo "Both z_start and z_end are required." >&2
    exit 2
  fi
  PREDICT_FLAG+=(--z-range "$Z_START" "$Z_END")
fi
if [ -n "$Z_REDUCTION" ]; then
  PREDICT_FLAG+=(--z-reduction "$Z_REDUCTION")
fi
if [ -n "$REFERENCE_PIXEL_SIZE" ]; then
  PREDICT_FLAG+=(--reference-pixel-size "$REFERENCE_PIXEL_SIZE")
fi
if [ -n "$REFERENCE_PIXEL_SIZE_Z_UM" ]; then
  PREDICT_FLAG+=(--reference-pixel-size-z-um "$REFERENCE_PIXEL_SIZE_Z_UM")
fi
if [ -n "$BATCH_SIZE" ]; then
  PREDICT_FLAG+=(--batch-size "$BATCH_SIZE")
fi
if [ -n "$Z_WINDOW" ]; then
  PREDICT_FLAG+=(--z-window "$Z_WINDOW")
fi
if [ -n "$FOCUS_CHANNEL" ]; then
  PREDICT_FLAG+=(--focus-channel "$FOCUS_CHANNEL")
fi
if [ -n "$Z_FOCUS_OFFSET" ]; then
  PREDICT_FLAG+=(--z-focus-offset "$Z_FOCUS_OFFSET")
fi

srun uv run --project "$WORKSPACE_DIR" dynaclr predict-batch \
  -c "$COLLECTION" \
  --model-family "$MODEL_FAMILY" \
  --run "$RUN" \
  --ckpt-name "$CKPT_NAME" \
  --datasets-root "$DATASETS_ROOT" \
  "${CKPT_FLAG[@]}" \
  "${MARKER_FLAG[@]}" \
  "${PREDICT_FLAG[@]}" \
  "${MODEL_TYPE_FLAG[@]}" \
  "${TRAINING_CONFIG_FLAG[@]}" \
  --num-workers 0
