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


@click.command()
@click.option("--m",    type=int, default = 12,  help="Exponent of subhalo mass.")
@click.option("--nsub", type=int, default = 1,   help="Number of subhaloes.")
@click.option("--nsim", type=int, default = 100, help="Number of simulations to run.")

@click.option("--lr",         type=float, default = 1e-3, help="Learning rate.")
@click.option("--factor",     type=float, default = 1e-1, help = "Factor of Scheduler")
@click.option("--patience",   type=int,   default = 5,    help = "Patience of Scheduler")
@click.option("--max_epochs", type=int,   default = 30,   help = "Max number of epochs.")



# m = 10
# nsub = 1
# nsim = 10000

# lr = 1e-3
# factor = 1e-1
# patience = 5
# max_epochs = 1


def run(m, nsub, nsim, lr, factor, patience, max_epochs):

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

    save_name, save_path = get_name(RUN, lr, factor, patience)
    print(f'Training {save_name}!')

    torch.set_default_tensor_type(torch.FloatTensor)
    post = swyft.Posteriors(dataset)
    post.add(marginals, device = DEVICE, head = CustomHead, tail = UNET)
    post.train(marginals, max_epochs = max_epochs,
               optimizer_args = dict(lr=lr),
               scheduler_args = dict(factor = factor, patience = patience)
              )

    post.save(save_path)
    print('Done!')
    
if __name__ == "__main__":
    run()
