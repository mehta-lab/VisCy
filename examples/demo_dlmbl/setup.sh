#!/usr/bin/env -S bash -i

START_DIR=$(pwd)

# Create conda environment
conda create -y --name 06_image_translation python=3.10

# Install ipykernel in the environment.
conda install -y ipykernel nbformat nbconvert black jupytext ipywidgets --name 06_image_translation
# Specifying the environment explicitly.
# conda activate sometimes doesn't work from within shell scripts.

# install viscy and its dependencies`s in the environment using pip.
mkdir -p ~/code/
cd ~/code/
git clone https://github.com/mehta-lab/viscy.git
cd viscy
git checkout main #FIXME: change after merging this PR   # Exercise is tested with this commit of viscy

# Find path to the environment - conda activate doesn't work from within shell scripts.
ENV_PATH=$(conda info --envs | grep 06_image_translation | awk '{print $NF}')
$ENV_PATH/bin/pip install ".[metrics]"

# Create the directory structure
mkdir -p ~/data/06_image_translation/training
mkdir -p ~/data/06_image_translation/test

# Change to the target directory
cd ~/data/06_image_translation/training

# Download the OME-Zarr dataset recursively
wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VSCyto2D/training/a549_hoechst_cellmask_train_val.zarr/"
cd ~/data/06_image_translation/test
wget -m -np -nH --cut-dirs=4 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/VSCyto2D/test/a549_hoechst_cellmask_test.zarr/"

# Change back to the starting directory
cd $START_DIR
