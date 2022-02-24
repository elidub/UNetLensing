#!/usr/bin/env python
# coding: utf-8

import os, datetime
import torch, pyro, numpy as np
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from torch import tensor
import torch.nn as nn
import torchvision.transforms.functional as TF

import click


import swyft

DEVICE = 'cuda'

from utils import *


@click.command()
@click.option("--m",    type=int, default = 12, help="Exponent of subhalo mass.")
@click.option("--nsub", type=int, default = 1, help="Number of subhaloes.")
@click.option("--nsim", type=int, default = 100, help="Number of simulations to run.")


def run(m, nsub, nsim):
    time_start = datetime.datetime.now()
    
    # Set definitions (should go to click)
    system_name = "ngc4414"
    
    # Set utilities
    sim_name, sim_path = get_sim_path(m, nsub, nsim, system_name)
    
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # HACK
    CONFIG = get_config(system_name, str(nsub), str(m))
    torch.set_default_tensor_type(torch.FloatTensor)
    
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # HACK
    ppd = CONFIG.ppd()['model_trace'].nodes
    torch.set_default_tensor_type(torch.FloatTensor)

    prior, uv, lows, highs = get_prior(CONFIG)
    L = CONFIG.kwargs["defs"]["nx"]
    print(f'Image has L = {L}.')

    assert nsub == CONFIG.umodel.alphas["main"].sub.nsub
    if m > 4:
        assert all([i == pow(10, m) for i in ppd['main/sub/m_sub']['value']])
    else:
        print(f'm = {m} <= 0!', ppd['main/sub/m_sub']['value'])


    # Create Store
    pnames = [f'{z}_{i+1}' for i in range(nsub) for z in ['x', 'y', 'm']]
    print(pnames)
    simulator = swyft.Simulator(model = lambda v: simul(v, CONFIG), 
                                pnames = pnames,
                                sim_shapes={"image": (L, L)})

    store = swyft.DirectoryStore(path = sim_path, simulator = simulator)
    store.add(nsim, prior)
    store.simulate()
    
    print('Done!')
    print(f"Total creating time is {str(datetime.datetime.now() - time_start).split('.')[0]}!")


if __name__ == "__main__":
    run()