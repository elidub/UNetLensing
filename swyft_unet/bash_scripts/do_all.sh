#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH -N 1
#SBATCH -p gpu_titanrtx_shared
#SBATCH --gpus-per-node=titanrtx:1

# cd /home/eliasd/lensing/pyrofit-lensing-analysis/experiments/swyft_masses
cd ../scripts
pwd

# python get_pred.py     --nsub 1 --nsim 1000 --sigma 0.0 --npred 0 --zero 'toy'


# python create_store.py --nsub 1 --nsim 500 --simul 'real'
# python train.py        --nsub 1 --nsim 500 --sigma 0.0 --simul 'real' --nmc 3
python get_pred.py     --nsub 1 --nsim 500 --sigma 0.0 --npred 0 --simul 'real' --nmc 3

# python create_store.py --nsub 2 --nsim 300 --simul 'toy'
# python train.py        --nsub 2 --nsim 300 --sigma 0.0 --simul 'toy'
# python get_pred.py     --nsub 2 --nsim 300 --sigma 0.0 --npred 0 --simul 'toy'

# python create_store.py --nsub 3 --nsim 50000 
# python train.py        --nsub 3 --nsim 50000 --sigma 0.0 
# python get_pred.py     --nsub 3 --nsim 50000 --sigma 0.0 --npred 0

# python create_store.py --nsub 5 --nsim 50000 
# python train.py        --nsub 5 --nsim 50000 --sigma 0.0 
# python get_pred.py     --nsub 5 --nsim 50000 --sigma 0.0 --npred 0




# python create_store.py --nsub 1 --nsim 10000 --simul 'toy'
# python train.py        --nsub 1 --nsim 10000 --sigma 0.0 --simul 'toy'
# python get_pred.py     --nsub 1 --nsim 10000 --sigma 0.0 --npred 0 --simul 'toy'

# python create_store.py --nsub 3 --nsim 10000 --simul 'toy'
# python train.py        --nsub 3 --nsim 10000 --sigma 0.0 --simul 'toy'
# python get_pred.py     --nsub 3 --nsim 10000 --sigma 0.0 --npred 0 --simul 'toy'

# python create_store.py --nsub 1 --nsim 30000 --simul 'toy'
# python train.py        --nsub 1 --nsim 30000 --sigma 0.0 --simul 'toy'
# python get_pred.py     --nsub 1 --nsim 30000 --sigma 0.0 --npred 0 --simul 'toy'



# python create_store.py --nsub 3 --nsim 50000

# python train.py        --nsub 3 --nsim 50000 --sigma 0.0 
# python get_pred.py     --nsub 3 --nsim 50000 --sigma 0.0 --npred 0

# python train.py        --nsub 3 --nsim 50000 --sigma 0.1 
# python get_pred.py     --nsub 3 --nsim 50000 --sigma 0.1 --npred 0

# python train.py        --nsub 3 --nsim 50000 --sigma 0.5 
# python get_pred.py     --nsub 3 --nsim 50000 --sigma 0.5 --npred 0



