#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH -N 1
#SBATCH -p gpu_titanrtx_shared
#SBATCH --gpus-per-node=titanrtx:1

pwd
cd ..
pwd

# python train.py --m 1 --nsub 3 --nsim 10000 --max_epochs 30 --lr 1e-3 --factor 1e-1