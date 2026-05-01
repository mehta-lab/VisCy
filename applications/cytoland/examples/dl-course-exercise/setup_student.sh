#!/usr/bin/env -S bash -i
#
# Image-translation exercise — STUDENT setup.
#
# This script:
#   1. Installs uv if missing (user-level, no sudo).
#   2. Creates a Python 3.13 venv under this folder (./.venv).
#   3. Installs cytoland + viscy (>=0.5.0a0) plus the tutorial extras:
#      cellpose, torchview, microssim, jupyter, ipywidgets, jupytext.
#      If run from inside a checkout of the VisCy monorepo, installs
#      the local cytoland workspace package in editable mode (pulls
#      viscy-data, viscy-models, viscy-transforms, viscy-utils from
#      the workspace). Otherwise installs from PyPI.
#   4. Registers the venv as a Jupyter kernel named "06_image_translation"
#      so students see it in VSCode / JupyterLab.
#   5. Downloads the training / test OME-Zarr datasets and the VSCyto2D
#      pretrained checkpoint into $DATA_ROOT (default ~/data/06_image_translation),
#      ONLY IF the data is not already there. If a TA has pre-staged data
#      on a shared filesystem, point DATA_ROOT at it to skip the download:
#
#        DATA_ROOT=/mnt/shared/image_translation bash setup_student.sh
#
# Run this from the exercise folder:
#   cd applications/cytoland/examples/dl-course-exercise
#   bash setup_student.sh

set -euo pipefail

START_DIR=$(pwd)
KERNEL_NAME="${KERNEL_NAME:-06_image_translation}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"

# --- Detect optional VisCy monorepo root (four levels up from this script) -
# When this exercise lives inside a viscy clone, install cytoland in editable
# mode against the local workspace. Otherwise fall back to PyPI.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONOREPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." 2>/dev/null && pwd || true)"
if [[ -n "${MONOREPO_ROOT:-}" && -f "$MONOREPO_ROOT/pyproject.toml" ]] \
        && grep -q '^name = "viscy"' "$MONOREPO_ROOT/pyproject.toml"; then
    INSTALL_MODE="workspace"
else
    INSTALL_MODE="pypi"
    MONOREPO_ROOT=""
fi
echo "Install mode: $INSTALL_MODE"

# --- 1. Install uv if missing ----------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — installing to ~/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer updates shell profiles but not the current shell
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "Using uv: $(uv --version)"

# --- 2. Create a venv under this exercise folder ---------------------------
VENV_DIR="$SCRIPT_DIR/.venv"
uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
PY="$VENV_DIR/bin/python"

# --- 3. Install cytoland + viscy + tutorial extras -------------------------
if [[ "$INSTALL_MODE" == "workspace" ]]; then
    echo "Installing cytoland (editable) from $MONOREPO_ROOT ..."
    uv pip install --python "$PY" -e "$MONOREPO_ROOT/applications/cytoland[metrics]"
else
    echo "Installing cytoland + viscy from PyPI (>=0.5.0a0) ..."
    uv pip install --python "$PY" --prerelease=allow \
        "viscy>=0.5.0a0" \
        "cytoland[metrics]>=0.5.0a0"
fi
uv pip install --python "$PY" \
    cellpose \
    torchview \
    microssim \
    jupyter \
    ipykernel \
    ipywidgets \
    jupytext \
    nbformat \
    nbconvert

# --- 4. Register the venv as a Jupyter kernel ------------------------------
"$PY" -m ipykernel install --user \
    --name "$KERNEL_NAME" \
    --display-name "Python ($KERNEL_NAME)"
echo "Registered Jupyter kernel: $KERNEL_NAME"

# --- 5. Download data + pretrained checkpoints (skip if already present) ----
DATA_ROOT="${DATA_ROOT:-$HOME/data/$KERNEL_NAME}"
TRAINING_ZARR="$DATA_ROOT/training/a549_hoechst_cellmask_train_val.zarr"
TEST_ZARR="$DATA_ROOT/test/a549_hoechst_cellmask_test.zarr"
CHECKPOINT="$DATA_ROOT/pretrained_models/VSCyto2D/epoch=399-step=23200.ckpt"
FLUOR2PHASE_CKPT="$DATA_ROOT/pretrained_models/AIMBL_Demo/fluor2phase_step668.ckpt"

mkdir -p "$DATA_ROOT/training" "$DATA_ROOT/test" "$DATA_ROOT/pretrained_models"

if [[ -d "$TRAINING_ZARR" && -d "$TEST_ZARR" && -f "$CHECKPOINT" && -f "$FLUOR2PHASE_CKPT" ]]; then
    echo "Data already present at $DATA_ROOT — skipping download."
else
    echo "Downloading data + checkpoints to $DATA_ROOT ..."
    cd "$DATA_ROOT/training"
    wget -m -np -nH --cut-dirs=6 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/training/zarrv3/a549_hoechst_cellmask_train_val.zarr/"

    cd "$DATA_ROOT/test"
    wget -m -np -nH --cut-dirs=6 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/test/zarrv3/a549_hoechst_cellmask_test.zarr/"

    cd "$DATA_ROOT/pretrained_models"
    wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto2D/VSCyto2D/epoch=399-step=23200.ckpt"
    # Second checkpoint used in Task 2.5 (fluorescence -> phase reverse model).
    wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto2D/AIMBL_Demo/fluor2phase_step668.ckpt"
fi

cd "$START_DIR"

cat <<EOF

--------------------------------------------------------------------
Student setup complete.

  - venv:           $VENV_DIR
  - jupyter kernel: $KERNEL_NAME
  - data:           $DATA_ROOT

To start the exercise:
  1. Launch Jupyter or open solution.py in VSCode.
  2. Select the "Python ($KERNEL_NAME)" kernel.
  3. Run cells top to bottom.
--------------------------------------------------------------------
EOF
