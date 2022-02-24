#!/usr/bin/env python
# coding: utf-8


import os
import torch, pyro, numpy as np
torch.set_default_tensor_type(torch.cuda.FloatTensor)

import swyft
import click


DEVICE = 'cuda'

from utils import *
from network import UNET, CustomHead

import optuna
import joblib


@click.command()
@click.option("--m",    type=int, default = 12,  help="Exponent of subhalo mass.")
@click.option("--nsub", type=int, default = 1,   help="Number of subhaloes.")
@click.option("--nsim", type=int, default = 100, help="Number of simulations to run.")

# @click.option("--lr",         type=float, default = 1e-3, help="Learning rate.")
# @click.option("--factor",     type=float, default = 1e-1, help = "Factor of Scheduler")
# @click.option("--patience",   type=int,   default = 5,    help = "Patience of Scheduler")
@click.option("--max_epochs", type=int,   default = 30,   help = "Max number of epochs.")


@click.option("--n_trials", type=int,   default = 5,   help = "Number of trials in grid search.")




def run(m, nsub, nsim, max_epochs, n_trials):

    SYSTEM_NAME = "ngc4414"
    RUN = f'_m{m}_nsub{nsub}_nsim{nsim}'
    assert os.path.exists(f'/nfs/scratch/eliasd/store{RUN}.sync')
    SIM_PATH = f'/nfs/scratch/eliasd/store{RUN}.zarr' 
    print('run', RUN)

    # Set utilities
    store = swyft.DirectoryStore(path=SIM_PATH)
    print(f'Store has {len(store)} simulations')

    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # HACK
    CONFIG = get_config(SYSTEM_NAME, str(nsub), str(m))
    torch.set_default_tensor_type(torch.FloatTensor)

    prior, uv = get_prior(CONFIG)


    # Set up posterior
    idx = 0
    img_0 = store[idx][0]['image']
    L1, L2 = torch.tensor(img_0.shape)
    assert L1 == L2
    L = L1.item()
    print(f'L = {L}')

    torch.set_default_tensor_type(torch.FloatTensor)
    dataset = swyft.Dataset(nsim, prior, store)#, simhook = noise)
    marginals = [i for i in range(L**2)]
    post = swyft.Posteriors(dataset)

    # Train
    
    def objective(trail):
        lr       = trail.suggest_float('lr', 1e-5, 1e-3, log = True)
        factor   = trail.suggest_float('factor', 1e-4, 1e-1, log = True)
        patience = trail.suggest_int('patience', 2, 5)

        save_name, save_path = get_name(RUN, lr, factor, patience, 'posts_gridsearch')
        print(f'Training {save_name}!')

        torch.set_default_tensor_type(torch.FloatTensor)
        post = swyft.Posteriors(dataset)
        post.add(marginals, device = DEVICE, head = CustomHead, tail = UNET)
        post.train(marginals, max_epochs = max_epochs,
                   optimizer_args = dict(lr=lr),
                   scheduler_args = dict(factor = factor, patience = patience)
                  )

        epoch, tl, vl = get_losses(post)
        post.save(save_path)
        
        print()

        return vl[-1]
    
    study_name = f'study{RUN}'
    storage_name = f'studies/{study_name}.pkl'
    
    if os.path.isfile(storage_name):
        print(f'{storage_name} is loaded!')
        study = joblib.load(storage_name)
    else:
        study = optuna.create_study(direction="minimize", study_name = study_name)
#                                     study_name = study_name, storage = storage_name, load_if_exists = True)
    
    study.optimize(objective, n_trials = n_trials)
    
    joblib.dump(study, storage_name)
    
    print('Done!')
    
if __name__ == "__main__":
    run()
