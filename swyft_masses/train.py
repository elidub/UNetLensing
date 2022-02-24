#!/usr/bin/env python
# coding: utf-8


import os, datetime
import torch, pyro, numpy as np
torch.set_default_tensor_type(torch.cuda.FloatTensor)

import swyft
import click


DEVICE = 'cuda'

from utils import *
from network import CustomTail, CustomHead


@click.command()
@click.option("--m",    type=int, default = 12,  help="Exponent of subhalo mass.")
@click.option("--nsub", type=int, default = 1,   help="Number of subhaloes.")
@click.option("--nsim", type=int, default = 100, help="Number of simulations to run.")

@click.option("--nmbins",  type=int, default = 2,   help="Number of mass bins.")

@click.option("--lr",         type=float, default = 1e-3, help="Learning rate.")
@click.option("--factor",     type=float, default = 1e-1, help = "Factor of Scheduler")
@click.option("--patience",   type=int,   default = 5,    help = "Patience of Scheduler")
@click.option("--max_epochs", type=int,   default = 30,   help = "Max number of epochs.")



# m = 10
# nsub = 1
# nsim = 10000

# nmbins = 2

# lr = 1e-3
# factor = 1e-1
# patience = 5
# max_epochs = 1


def run(m, nsub, nsim, nmbins, lr, factor, patience, max_epochs):
    time_start = datetime.datetime.now()
    
    # Set definitions (should go to click)
    system_name = "ngc4414"
    
    # Set utilities
    sim_name, sim_path = get_sim_path(m, nsub, nsim, system_name)
    store = swyft.DirectoryStore(path=sim_path)
    print(f'Store has {len(store)} simulations.')
    
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # HACK
    CONFIG = get_config(system_name, str(nsub), str(m))
    torch.set_default_tensor_type(torch.FloatTensor)

    prior, uv, lows, highs = get_prior(CONFIG)
    L = CONFIG.kwargs["defs"]["nx"]
    print(f'Image has L = {L}.')

    
    # Set up posterior
    torch.set_default_tensor_type(torch.FloatTensor)
    dataset = swyft.Dataset(nsim, prior, store)#, simhook = noise)
    marginals = [i for i in range(L**2)]
    post = swyft.Posteriors(dataset)

    
    # Train
    post_name, post_path = get_post_path(sim_name, nmbins, lr, factor, patience)
    print(f'Training {post_name}!')

    torch.set_default_tensor_type(torch.FloatTensor)
    post = swyft.Posteriors(dataset)
    post.add(marginals, device = DEVICE, 
             tail_args = dict(nmbins = nmbins, lows = lows, highs = highs),
             head = CustomHead, tail = CustomTail)
    post.train(marginals, max_epochs = max_epochs,
               optimizer_args = dict(lr=lr),
               scheduler_args = dict(factor = factor, patience = patience)
              )
    post.save(post_path)
    
    print('Done!')
    print(f"Total training time is {str(datetime.datetime.now() - time_start).split('.')[0]}!")
    
if __name__ == "__main__":
    run()
