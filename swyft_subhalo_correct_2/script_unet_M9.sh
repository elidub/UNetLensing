#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH -p gpu_titanrtx_shared
#SBATCH --gpus-per-node=titanrtx:1

cd /home/eliasd/lensing/pyrofit-lensing-analysis/experiments/swyft_subhalo_correct_2
python swyft_subhalo_unet_M9.py $1