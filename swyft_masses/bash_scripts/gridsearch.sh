#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH -N 1
#SBATCH -p gpu_titanrtx_shared
#SBATCH --gpus-per-node=titanrtx:1

cd /home/eliasd/lensing/pyrofit-lensing-analysis/experiments/swyft_subhalo_3
python gridsearch.py --m 0 --nsub 3 --nsim 50000 --max_epochs 30 --n_trials 3