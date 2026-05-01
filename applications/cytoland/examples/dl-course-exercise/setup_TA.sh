#!/usr/bin/env -S bash -i
#
# Image-translation exercise — TA / course-operator setup.
#
# Pre-stage the OME-Zarr datasets and pretrained VSCyto2D checkpoint onto a
# shared filesystem BEFORE the course starts so each student doesn't have to
# re-download ~14 GB. This typically takes 20–40 min depending on link speed
# and storage backend.
#
# Usage:
#
#   # Default: stage to ~/data/image_translation/
#   bash setup_TA.sh
#
#   # Stage to a shared mount (recommended for courses):
#   DATA_ROOT=/mnt/efs/image_translation bash setup_TA.sh
#
# Once this finishes, students point setup_student.sh at the same DATA_ROOT
# and skip the download:
#
#   DATA_ROOT=/mnt/efs/image_translation bash setup_student.sh
#
# This script does NOT create a Python environment. Run setup_student.sh for
# that (it can be run before, after, or instead of this script).

set -euo pipefail

START_DIR=$(pwd)
KERNEL_NAME="${KERNEL_NAME:-06_image_translation}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/$KERNEL_NAME}"

mkdir -p "$DATA_ROOT/training" "$DATA_ROOT/test" "$DATA_ROOT/pretrained_models"

echo "Staging data + checkpoint into $DATA_ROOT ..."
echo "(this typically takes 20-40 min)"

cd "$DATA_ROOT/training"
wget -m -np -nH --cut-dirs=6 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/training/zarrv3/a549_hoechst_cellmask_train_val.zarr/"

cd "$DATA_ROOT/test"
wget -m -np -nH --cut-dirs=6 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/test/zarrv3/a549_hoechst_cellmask_test.zarr/"

cd "$DATA_ROOT/pretrained_models"
wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto2D/VSCyto2D/epoch=399-step=23200.ckpt"
# Optional second checkpoint used in the AIMBL_Demo (fluor->phase). Uncomment
# to also stage it:
# wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto2D/AIMBL_Demo/fluor2phase_step668.ckpt"

cd "$START_DIR"

cat <<EOF

--------------------------------------------------------------------
TA setup complete.

  - data: $DATA_ROOT

Tell students to run:
  DATA_ROOT=$DATA_ROOT bash setup_student.sh

This will create their per-user venv + jupyter kernel and reuse the
pre-staged data (no re-download).
--------------------------------------------------------------------
EOF
