#!/usr/bin/env python
# coding: utf-8


import torch, datetime, click

import swyft
from utils import *
from data_mgmt import get_paths

DEVICE = 'cuda'


# from swyft.utils import tensor_to_array, array_to_tensor
# from toolz import compose
# from pyrofit.lensing.distributions import get_default_shmf


@click.command()
@click.option("--m",    type=int, default = 1, help="Exponent of subhalo mass.")
@click.option("--nsub", type=int, default = 1, help="Number of subhaloes.")
@click.option("--nsim", type=int, default = 100, help="Number of simulations to run.")

@click.option("--simul", type=str, default = 'real', help="What kind of simulator")


# m = 0
# nsub = 3
# nsim = 200


def run(m, nsub, nsim, simul):
    time_start = datetime.datetime.now()
    
    # Set definitions (should go to click)
    systemname = "ngc4414"
    
    # Initialize
    store_path, _, _, _ = get_paths(dict(m=m,nsub=nsub,nsim=nsim,simul=simul))
   
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # HACK
    config = get_config(systemname, nsub, m)
    torch.set_default_tensor_type(torch.FloatTensor)
    
#     torch.set_default_tensor_type(torch.cuda.FloatTensor)  # HACK
#     ppd = config.ppd()['model_trace'].nodes
#     torch.set_default_tensor_type(torch.FloatTensor)

    prior, n_pars, lows, highs = get_prior(config)
    L = config.kwargs["defs"]["nx"]
#     print(f'Image has L = {L}.')

#     assert nsub == config.umodel.alphas["main"].sub.nsub
#     print('m samples:', [f"{i:.2}" for i in ppd['main/sub/m_sub']['value']])
    
    if simul == 'toy':
        simul_choose = simul_toy
    else:
        simul_choose = simul_lens


    # Create Store
    simulator = swyft.Simulator(model = lambda v: simul_choose(v, config), 
                                parameter_names = n_pars,
                                sim_shapes={"image": (L, L)})
    store = swyft.Store.directory_store(
        overwrite = True, path = store_path, 
        simulator = simulator)
    store.add(nsim, prior)
    store.simulate()

    print_duration(f'Creating store {store_path}', time_start)


if __name__ == "__main__":
    run()