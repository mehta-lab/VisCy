#!/usr/bin/env -S bash -i

START_DIR=$(pwd)

conda deactivate
# Create conda environment
conda create -y --name vs_Phc python=3.11

# Install ipykernel in the environment.
conda install -y ipykernel nbformat nbconvert black jupytext ipywidgets --name vs_Phc
# Specifying the environment explicitly.
# conda activate sometimes doesn't work from within shell scripts.

# install viscy and its dependencies`s in the environment using pip.
# Find path to the environment - conda activate doesn't work from within shell scripts.
ENV_PATH=$(conda info --envs | grep vs_Phc | awk '{print $NF}')
$ENV_PATH/bin/pip install "viscy[metrics,visual,examples]>=0.2"

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
