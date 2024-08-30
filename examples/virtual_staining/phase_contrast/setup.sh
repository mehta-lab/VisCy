#!/usr/bin/env -S bash -i

START_DIR=$(pwd)

conda deactivate
# Create conda environment
conda create -y --name vs_Phc python=3.10

# Install ipykernel in the environment.
conda install -y ipykernel nbformat nbconvert black jupytext ipywidgets --name vs_Phc
# Specifying the environment explicitly.
# conda activate sometimes doesn't work from within shell scripts.

# install viscy and its dependencies`s in the environment using pip.
# Find path to the environment - conda activate doesn't work from within shell scripts.
ENV_PATH=$(conda info --envs | grep vs_Phc | awk '{print $NF}')
$ENV_PATH/bin/pip install "viscy[metrics,visual]==0.2.1"

# Create the directory structure
mkdir -p ~/data/vs_PhC/training
mkdir -p ~/data/vs_PhC/test

# Change to the target directory
cd ~/data/vs_PhC/training
# Download the OME-Zarr dataset recursively
wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VSCyto2D/training/a549_hoechst_cellmask_train_val.zarr/"
cd ~/data/vs_PhC/test
wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VSCyto2D/test/a549_hoechst_cellmask_test.zarr/"

# Change back to the starting directory
cd $START_DIR
