#!/usr/bin/env -S bash -i

START_DIR=$(pwd)

conda deactivate
# Create conda environment
conda create -y --name image2image python=3.11

# Install ipykernel in the environment.
conda install -y ipykernel nbformat nbconvert black jupytext ipywidgets --name img2img
# Specifying the environment explicitly.
# conda activate sometimes doesn't work from within shell scripts.

# install viscy and its dependencies`s in the environment using pip.
# Find path to the environment - conda activate doesn't work from within shell scripts.
ENV_PATH=$(conda info --envs | grep img2img | awk '{print $NF}')
$ENV_PATH/bin/pip install "viscy[metrics,visual,examples]>=0.2"

# Create the directory structure
mkdir -p ~/data/img2img/training
mkdir -p ~/data/img2img/test

# Change to the target directory
cd ~/data/img2img/training
# Download the OME-Zarr dataset recursively
wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VSCyto2D/training/a549_hoechst_cellmask_train_val.zarr/"
cd ~/data/img2img/test
wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VSCyto2D/test/a549_hoechst_cellmask_test.zarr/"

# Change back to the starting directory
cd $START_DIR
