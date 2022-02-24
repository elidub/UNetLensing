#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH -N 1
#SBATCH -p gpu_titanrtx_shared
#SBATCH --gpus-per-node=titanrtx:1

# cd /home/eliasd/lensing/pyrofit-lensing-analysis/experiments/swyft_masses
cd ../scripts
pwd

python train.py --m 1 --nsub 5 --nsim 50000 --nmbins 3 --sigma 0. --max_epochs 20