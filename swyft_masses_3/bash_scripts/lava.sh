#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH -p gpu_titanrtx_shared
#SBATCH --gpus-per-node=titanrtx:1

# cd /home/eliasd/lensing/pyrofit-lensing-analysis/experiments/swyft_masses
cd ../scripts
pwd

python lava.py --m 1 --nsub 3 --nsim 25000 --nmbins 10 --n_plot 3