#!/usr/bin/env bash

START_DIR=$(pwd)

# Create the directory structure
output_dir=~/
mkdir -p "$output_dir"/data/dynaclr/demo

# Change to the target directory if you want to download the data to a specific directory
cd ~/data/dynaclr/demo

# Download the data
wget -m -np -nH --cut-dirs=6 -R "index.html*" "https://public.czbiohub.org/comp.micro/viscy/DynaCLR_data/DENV/test/20240204_A549_DENV_ZIKV_timelapse/"

echo "Data downloaded successfully."

# Change back to the starting directory
cd $START_DIR
