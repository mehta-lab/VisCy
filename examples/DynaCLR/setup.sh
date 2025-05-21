#!/usr/bin/env bash

START_DIR=$(pwd)

# Initialize conda for the shell
eval "$(conda shell.bash hook)"
conda deactivate

# Check if environment exists
if ! conda env list | grep -q "dynaclr"; then
    echo "Creating new dynaclr environment..."
    conda config --add channels defaults
    conda create -y --name dynaclr python=3.11
else
    echo "Environment already exists. Updating packages..."
fi

# Activate the environment
conda activate dynaclr


# Install/update conda packages
conda install -y ipykernel nbformat nbconvert black jupytext ipywidgets
python -m ipykernel install --user --name dynaclr --display-name "Python (dynaclr)"

# Install viscy and its dependencies using pip
pip install "viscy[visual,metrics] @ ./viscy-dynaclr_v2.zip"
pip install gdown

# Download data from Google Drive
gdown    -O dynaclr_demo_data.tar.gz

# Unzip the data tar.gz file
tar -xzf dynaclr_demo_data.tar.gz -C ./dynaclr_demo_data

# Deactivate the environment
conda deactivate

# Change back to the starting directory
cd $START_DIR

echo "DynaClr environment setup complete."
