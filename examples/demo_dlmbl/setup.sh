#!/usr/bin/env -S bash -i

START_DIR=$(pwd)

# Create mamba environment
mamba create -y --name 04_image_translation python=3.10

# Install ipykernel in the environment.
mamba install -y ipykernel nbformat nbconvert black jupytext ipywidgets --name 04_image_translation
# Specifying the environment explicitly.
# mamba activate sometimes doesn't work from within shell scripts.

# install viscy and its dependencies`s in the environment using pip.
mkdir -p ~/code/
cd ~/code/
git clone https://github.com/mehta-lab/viscy.git
cd viscy
git checkout 7c5e4c1d68e70163cf514d22c475da8ea7dc3a88 # Exercise is tested with this commit of viscy
# Find path to the environment - mamba activate doesn't work from within shell scripts.
ENV_PATH=$(conda info --envs | grep 04_image_translation | awk '{print $NF}')
$ENV_PATH/bin/pip install ".[metrics]"

# Create data directory
mkdir -p ~/data/04_image_translation
cd ~/data/04_image_translation
wget https://dl-at-mbl-2023-data.s3.us-east-2.amazonaws.com/DLMBL2023_image_translation_data_pyramid.tar.gz
wget https://dl-at-mbl-2023-data.s3.us-east-2.amazonaws.com/DLMBL2023_image_translation_test.tar.gz
tar -xzf DLMBL2023_image_translation_data_pyramid.tar.gz
tar -xzf DLMBL2023_image_translation_test.tar.gz

# Change back to the starting directory
cd $START_DIR
