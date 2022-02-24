#!/usr/bin/env python
# coding: utf-8



import torch, datetime, click
torch.set_default_tensor_type(torch.cuda.FloatTensor)

import swyft
from utils import *
from data_mgmt import get_paths


DEVICE = 'cuda'


@click.command()
@click.option("--m",    type=int, default = 1,  help="Exponent of subhalo mass.")
@click.option("--nsub", type=int, default = 1,   help="Number of subhaloes.")
@click.option("--nsim", type=int, default = 100, help="Number of simulations to run.")

@click.option("--nmc",  type=int, default = 1,   help="Number of mass bins.")
@click.option("--sigma",   type=float, default = 0.0,   help="Additional noise.")


@click.option("--lr",         type=float, default = 1e-3, help="Learning rate.")
@click.option("--factor",     type=float, default = 1e-1, help = "Factor of Scheduler")
@click.option("--patience",   type=int,   default = 5,    help = "Patience of Scheduler")
@click.option("--max_epochs", type=int,   default = 20,   help = "Max number of epochs.")

@click.option("--simul", type=str, default = 'real', help="Number of simulations to run.")


# m = 0
# nsub = 3
# nsim = 200

# nmbins = 2

# lr = 1e-3
# factor = 1e-1
# patience = 5
# max_epochs = 1


def run(m, nsub, nsim, nmc, sigma, lr, factor, patience, max_epochs, simul):
    time_start = datetime.datetime.now()
    
    def noise(obs, _= None, sigma_n = sigma):
        image = obs["image"]
        eps = np.random.randn(*image.shape) * sigma_n
        return {"image": image + eps}
    
    # Set definitions (should go to click)
    systemname = "ngc4414"
    
    # Set utilities
    
    store_path, dataset_path, mre_path, _ = get_paths(dict(m=m,nsub=nsub,nsim=nsim,nmc=nmc,sigma=sigma,simul=simul))
    
    store = swyft.Store.load(path=store_path)
    print(f'Store has {len(store)} simulations.')
    
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # HACK
    config = get_config(systemname, str(nsub), str(m))
    torch.set_default_tensor_type(torch.FloatTensor)

    prior, n_pars, lows, highs = get_prior(config)
    L = config.kwargs["defs"]["nx"]
#     print(f'Image has L = {L}.')
    
    dataset = swyft.Dataset(nsim, prior, store, simhook = noise)

    # Train network
    print(f'Training {mre_path}!')
    
    marginal_indices, _ = swyft.utils.get_corner_marginal_indices(n_pars)

    network = get_custom_marginal_classifier(
        observation_transform = CustomObservationTransform('image', {'image': (L, L)}),
        marginal_indices = marginal_indices,
#         L = L,
        nmc = nmc, 
#         lows = lows,
#         highs = highs,
        marginal_classifier = CustomMarginalClassifier,
        parameter_transform = CustomParameterTransform(nmc, L, lows, highs)
    )

    mre = swyft.MarginalRatioEstimator(
        marginal_indices = marginal_indices,
        network = network,
        device = DEVICE,
    )

    _ = mre.train(dataset, max_epochs = max_epochs)

    mre.save(mre_path)
    dataset.save(dataset_path)
    
    print_duration(f'Training {mre_path}', time_start)

    
if __name__ == "__main__":
    run()
