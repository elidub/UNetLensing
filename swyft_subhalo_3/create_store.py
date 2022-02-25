#!/usr/bin/env python
# coding: utf-8

import os
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
@click.option("--nsim", type=int, default=100, help="Number of simulations to run.")


# In[ ]:

def run(m, nsub, nsim):


    SYSTEM_NAME = "ngc4414"
    RUN = f'_m{m}_nsub{nsub}_nsim{nsim}'
    SIM_PATH = f'/nfs/scratch/eliasd/store{RUN}.zarr' 
    print('run', RUN)
    

    # Set utilities
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # HACK
    CONFIG = get_config(SYSTEM_NAME, str(nsub), str(m))
    torch.set_default_tensor_type(torch.FloatTensor)

    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # HACK
    ppd = CONFIG.ppd()['model_trace'].nodes
    torch.set_default_tensor_type(torch.FloatTensor)


    # Check simulation
    v = ppd['main/sub/p_sub']['value']
    sim = simul(v, CONFIG)

    print('v', v)
    print('sim', sim)

    print("sim['image'] min and max" , sim['image'].min(), sim['image'].max() )


    prior, uv = get_prior(CONFIG)
    nx = CONFIG.kwargs["defs"]["nx"]
    ny = CONFIG.kwargs["defs"]["ny"]
    
    assert nsub == CONFIG.umodel.alphas["main"].sub.nsub
    if m > 4:
        assert all([i == pow(10, m) for i in ppd['main/sub/m_sub']['value']])
    else:
        print(f'm = {m} <= 0!', ppd['main/sub/m_sub']['value'])


    # Create Store
    pnames = [f'{z}_{i+1}' for i in range(nsub) for z in ['x', 'y']]
    print(pnames)
    simulator = swyft.Simulator(model = lambda v: simul(v, CONFIG), 
    #                             pnames = ["x_sub", "y_sub"],
                                pnames = pnames,
                                sim_shapes={"image": (nx, ny)})

    store = swyft.DirectoryStore(path=SIM_PATH, simulator=simulator)

    store.add(nsim, prior)
    store.simulate()
    
    print('Done!')


if __name__ == "__main__":
    run()