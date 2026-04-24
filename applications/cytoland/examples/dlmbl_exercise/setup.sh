#!/usr/bin/env -S bash -i
#
# DL@MBL image-translation exercise — environment + data setup.
#
# This script:
#   1. Installs uv if missing (user-level, no sudo).
#   2. Creates a Python 3.11 venv under this folder (./.venv).
#   3. Installs cytoland (from the VisCy monorepo) plus the tutorial
#      extras: cellpose, torchview, jupyter, ipywidgets, jupytext.
#   4. Registers the venv as a Jupyter kernel named "06_image_translation"
#      so students see it in VSCode / JupyterLab.
#   5. Downloads the training / test OME-Zarr datasets and the VSCyto2D
#      pretrained checkpoint into ~/data/06_image_translation/.
#
# Run this from the exercise folder:
#   cd applications/cytoland/examples/dlmbl_exercise
#   bash setup.sh

set -euo pipefail

START_DIR=$(pwd)
KERNEL_NAME="06_image_translation"
PYTHON_VERSION="3.11"

# --- Resolve VisCy monorepo root (two levels up from this script) -----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONOREPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
if [[ ! -f "$MONOREPO_ROOT/pyproject.toml" ]]; then
    echo "Could not find VisCy monorepo root at $MONOREPO_ROOT" >&2
    exit 1
fi

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

# --- 3. Install cytoland + tutorial extras ---------------------------------
uv pip install --python "$PY" -e "$MONOREPO_ROOT/applications/cytoland[metrics]"
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

# --- 5. Download data + pretrained checkpoint ------------------------------
DATA_ROOT="$HOME/data/$KERNEL_NAME"
mkdir -p "$DATA_ROOT/training" "$DATA_ROOT/test" "$DATA_ROOT/pretrained_models"

cd "$DATA_ROOT/training"
wget -m -np -nH --cut-dirs=5 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/training/a549_hoechst_cellmask_train_val.zarr/"

cd "$DATA_ROOT/test"
wget -m -np -nH --cut-dirs=5 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/test/a549_hoechst_cellmask_test.zarr/"

cd "$DATA_ROOT/pretrained_models"
wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto2D/VSCyto2D/epoch=399-step=23200.ckpt"

cd "$START_DIR"

cat <<EOF

--------------------------------------------------------------------
Setup complete.

  - venv: $VENV_DIR
  - jupyter kernel: $KERNEL_NAME
  - data: $DATA_ROOT

To start the exercise:
  1. Launch Jupyter or open solution.py in VSCode.
  2. Select the "Python ($KERNEL_NAME)" kernel.
  3. Run cells top to bottom.
--------------------------------------------------------------------
EOF
