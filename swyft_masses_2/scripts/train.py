#!/usr/bin/env python
# coding: utf-8



import torch, datetime, click
torch.set_default_tensor_type(torch.cuda.FloatTensor)

import swyft
from utils import *

DEVICE = 'cuda'


@click.command()
@click.option("--m",    type=int, default = 12,  help="Exponent of subhalo mass.")
@click.option("--nsub", type=int, default = 1,   help="Number of subhaloes.")
@click.option("--nsim", type=int, default = 100, help="Number of simulations to run.")

@click.option("--nmbins",  type=int, default = 2,   help="Number of mass bins.")
@click.option("--sigma",   type=float, default = 0.,   help="Additional noise.")


@click.option("--lr",         type=float, default = 1e-3, help="Learning rate.")
@click.option("--factor",     type=float, default = 1e-1, help = "Factor of Scheduler")
@click.option("--patience",   type=int,   default = 5,    help = "Patience of Scheduler")
@click.option("--max_epochs", type=int,   default = 30,   help = "Max number of epochs.")



# m = 0
# nsub = 3
# nsim = 200

# nmbins = 2

# lr = 1e-3
# factor = 1e-1
# patience = 5
# max_epochs = 1


def run(m, nsub, nsim, nmbins, sigma, lr, factor, patience, max_epochs):
    time_start = datetime.datetime.now()
    
    def noise(obs, _= None, sigma_n = sigma):
        image = obs["image"]
        eps = np.random.randn(*image.shape) * sigma_n
        return {"image": image + eps}
    
    # Set definitions (should go to click)
    system_name = "ngc4414"
    
    # Set utilities
    sim_name, sim_path = get_sim_path(m, nsub, nsim, system_name)
    store = swyft.Store.load(path=sim_path)
    print(f'Store has {len(store)} simulations.')
    
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # HACK
    config = get_config(system_name, str(nsub), str(m))
    torch.set_default_tensor_type(torch.FloatTensor)

    prior, n_pars, lows, highs = get_prior(config)
    L = config.kwargs["defs"]["nx"]
    print(f'Image has L = {L}.')
    
    dataset_name, dataset_path = get_dataset_path(m, nsub, nsim, system_name, sigma)
    dataset = swyft.Dataset(nsim, prior, store, simhook = noise)

    # Train network
    mre_name, mre_path = get_mre_path(sim_name, nmbins, sigma, lr, factor, patience)
    print(f'Training {mre_name}!')
    
    marginal_indices, _ = swyft.utils.get_corner_marginal_indices(n_pars)

    network = get_custom_marginal_classifier(
        observation_transform = CustomObservationTransform('image', {'image': (L, L)}),
        marginal_indices = marginal_indices,
        L = L,
        nmbins = nmbins, 
        lows = lows,
        highs = highs,
        marginal_classifier = CustomMarginalClassifier,
    )

    mre = swyft.MarginalRatioEstimator(
        marginal_indices = marginal_indices,
        network = network,
        device = DEVICE,
    )

    _ = mre.train(dataset, max_epochs = max_epochs)

    mre.save(mre_path)
    dataset.save(dataset_path)
    
    
    print('Done!')
    print(f"Total training time is {str(datetime.datetime.now() - time_start).split('.')[0]}!")
    
if __name__ == "__main__":
    run()
