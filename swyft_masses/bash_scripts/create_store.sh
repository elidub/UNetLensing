#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH -p gpu_titanrtx_shared
#SBATCH --gpus-per-node=titanrtx:1

# cd /home/eliasd/lensing/pyrofit-lensing-analysis/experiments/swyft_masses
cd ..
pwd

python create_store.py --m 0 --nsub 10 --nsim 50000