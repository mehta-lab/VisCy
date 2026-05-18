#!/usr/bin/env -S bash -i

START_DIR=$(pwd)

# Resolve this script's directory so install paths work regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Cytoland package lives two levels up from this examples folder
# (applications/cytoland/examples/phase_contrast -> applications/cytoland).
CYTOLAND_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

conda deactivate
# Create conda environment
conda create -y --name vs_Phc python=3.12

# Install Jupyter kernel, notebook server, and related tooling in the environment.
conda install -y ipykernel notebook nbformat nbconvert ruff jupytext ipywidgets --name vs_Phc
# Specifying the environment explicitly.
# conda activate sometimes doesn't work from within shell scripts.

# Install cytoland (pulls in viscy-data, viscy-models, viscy-transforms, viscy-utils).
# Find path to the environment - conda activate doesn't work from within shell scripts.
ENV_PATH=$(conda info --envs | grep vs_Phc | awk '{print $NF}')
$ENV_PATH/bin/pip install -e "${CYTOLAND_DIR}[metrics]"

# Create the directory structure
mkdir -p ~/data/vs_PhC/test
mkdir -p ~/data/vs_PhC/models

# Change to the target directory
# Download the OME-Zarr dataset recursively
cd ~/data/vs_PhC/test
wget -m -np -nH --cut-dirs=5 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto3D/test/HEK_H2B_CAAX_PhC_40x_registered.zarr/"

# Get the models
cd ~/data/vs_PhC/models
wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto3D/no_augmentations/best_epoch=30-step=6076.ckpt"
wget -m -np -nH --cut-dirs=5 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto3D/epoch=48-step=18130.ckpt"


# Change back to the starting directory
cd $START_DIR
