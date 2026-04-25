#!/bin/bash
# CTC tracking accuracy benchmark — DynaCLR-2D-MIP vs IoU baseline
# Runs on all 9 2D CTC training datasets.
#
# sbatch applications/dynaclr/configs/evaluation/ctc_tracking_2d_mip_boc_all.sh

#SBATCH --job-name=ctc_tracking
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --output=%x-%j.out

export PYTHONNOUSERSITE=1
export GRB_LICENSE_FILE=/home/eduardo.hirata/gurobi/gurobi.lic

WORKSPACE="/hpc/mydata/eduardo.hirata/repos/viscy"
CONFIG="$WORKSPACE/applications/dynaclr/configs/evaluation/ctc_tracking_2d_mip_boc_all.yaml"

uv run --project "$WORKSPACE" dynaclr evaluate-tracking-accuracy -c "$CONFIG"
