#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH -N 1
#SBATCH -p gpu_titanrtx_shared
#SBATCH --gpus-per-node=titanrtx:1

# cd /home/eliasd/lensing/pyrofit-lensing-analysis/experiments/swyft_masses
cd ..
pwd

python train.py --m 0 --nsub 10 --nsim 50000 --nmbins 4 --max_epochs 30 --lr 1e-3 --factor 1e-1