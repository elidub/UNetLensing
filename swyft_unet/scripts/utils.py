import os, torch, pyro, datetime, numpy as np

torch.set_default_tensor_type(torch.cuda.FloatTensor)  # HACK

from clipppy import load_config, Clipppy
from clipppy.patches import torch_numpy
from ruamel.yaml import YAML

import swyft
from pyrofit.lensing.distributions import get_default_shmf
from classifier import *

DEVICE = 'cuda'

def print_duration(activity, time_start):
    print(f"{activity} Done!")
#     if saved not None: print(f"Saved {saved}!")
    print(f"Total creating time is {str(datetime.datetime.now() - time_start).split('.')[0]}! \n \n ")

def get_custom_marginal_classifier(
    observation_transform,
    marginal_indices: tuple,
    nsub, nmc, lows, highs, L,
#     nmc, 
#     lows,
#     highs,
    marginal_classifier,
    parameter_transform
) -> torch.nn.Module:
    
    n_observation_features = observation_transform.n_features
    
    marginal_classifier = marginal_classifier(
        len(marginal_indices),
        n_observation_features,
        nsub, nmc, lows, highs, L,
    )

    return swyft.networks.Network(
        observation_transform,
        parameter_transform,
        marginal_classifier,
    )

def get_config(system_name: str, nsub, m) -> Clipppy:
    """
    Get configuration
    """
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # HACK

    SOURCE_DIR = '../../../mock_data/sources'
        
    source_name = f'{system_name}.npy'
    config = load_config(f'../configs/config_nsub{str(nsub)}_m{str(m)}.yaml', base_dir=SOURCE_DIR)

    torch.set_default_tensor_type(torch.FloatTensor)  # HACK
    return config


def get_prior(CONFIG):
    config = CONFIG
    main = config.umodel.alphas["main"]
    prior_p_sub = main.sub.pos_sampler.base_dist
    m_sub_grid = main.sub.mass_sampler.y

    nsub = main.sub.nsub
    z_lens = config.kwargs['defs']['z_lens']

    lows = np.array([
            prior_p_sub.low[0].item(),
            prior_p_sub.low[1].item(),
            m_sub_grid.min().log10().item(),
        ])
    highs = np.array([
            prior_p_sub.high[0].item(),
            prior_p_sub.high[1].item(),
            m_sub_grid.max().log10().item(),
        ])
    
    uniform = torch.distributions.Uniform(torch.tensor(lows[:-1]), torch.tensor(highs[:-1]))
    shmf = get_default_shmf(z_lens = z_lens, log_range = (lows[-1], highs[-1]))

    parameter_dimensions = [2, 1]*nsub
    n_pars = sum(parameter_dimensions)
    
    prior = swyft.Prior.composite_prior(
        cdfs=list(map(swyft.Prior.conjugate_tensor_func, [uniform.cdf, shmf.cdf]*nsub)),
        icdfs=list(map(swyft.Prior.conjugate_tensor_func, [uniform.icdf, shmf.icdf]*nsub)),
        log_probs=list(map(swyft.Prior.conjugate_tensor_func, [uniform.log_prob, shmf.log_prob]*nsub)),
        parameter_dimensions=parameter_dimensions,
    )

    return prior, n_pars, lows, highs


def simul_lens(v, config: Clipppy):
    """
    Fix values for main lens and source parameters from config and put
    in a subhalo with the specified position and mass.

    Arguments
    - v: array containing x_sub, y_sub.

    Returns
    - Numpy array.
    """
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # HACK
    
    from pyrofit.lensing.utils import get_meshgrid  # import here due to HACKs
    nx = config.kwargs["defs"]["nx"]
    ny = config.kwargs["defs"]["ny"]
    res = config.kwargs["defs"]["res"]
    nsub = config.umodel.alphas["main"].sub.nsub
    X, Y = config.umodel.X.clone(), config.umodel.Y.clone()
    # Upsample image
    upsample = 10
    config.umodel.coerce_XY(*get_meshgrid(res / upsample, nx * upsample, ny * upsample))
        
    if not torch.is_tensor(v):
        v = torch.tensor(v)

    x_sub, y_sub, m_sub = v.view(-1,3).T.to(DEVICE)
    xy_sub = torch.stack((x_sub, y_sub)).T
    
    d_m_sub = pyro.distributions.Delta(m_sub)
    d_p_sub = pyro.distributions.Delta(xy_sub).to_event(1)


    def _guide():
        # Sample subhalo position
        guide_sample = {
            "main/sub/p_sub": pyro.sample("main/sub/p_sub", d_p_sub),
            "main/sub/m_sub": pyro.sample("main/sub/m_sub", d_m_sub),
        }

        return guide_sample
    
    result = {
        "image": config.ppd(guide=_guide)["model_trace"]
        .nodes["mu"]["value"]
        .detach()
        .numpy()
    }
    
    # Restore coarse grid
    config.umodel.coerce_XY(X, Y)
    # Downsample image
    averager = torch.nn.AvgPool2d((upsample, upsample))
    result['image'] = (averager(torch.tensor(result['image']).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0))

    torch.set_default_tensor_type(torch.FloatTensor)  # HACK
    return result

def simul_toy(v, config: Clipppy):
    """
    Fix values for main lens and source parameters from config and put
    in a subhalo with the specified position and mass.

    Arguments
    - v: array containing x_sub, y_sub.

    Returns
    - Numpy array.
    """
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # HACK
    
    from pyrofit.lensing.utils import get_meshgrid  # import here due to HACKs
    nx = config.kwargs["defs"]["nx"]
    ny = config.kwargs["defs"]["ny"]
    res = config.kwargs["defs"]["res"]
    nsub = config.umodel.alphas["main"].sub.nsub
    X, Y = config.umodel.X.clone(), config.umodel.Y.clone()
    # Upsample image
    upsample = 10
    config.umodel.coerce_XY(*get_meshgrid(res / upsample, nx * upsample, ny * upsample))
        
    if not torch.is_tensor(v):
        v = torch.tensor(v)

    x_sub, y_sub, m_sub = v.view(-1,3).T.to(DEVICE)
    xy_sub = torch.stack((x_sub, y_sub)).T
    
    L = 40
    grid = torch.linspace(-2.5, 2.5, L, device = DEVICE)
    Xs, Ys = torch.meshgrid(grid, grid, indexing = 'ij')
    
    w_sub = torch.tensor((0.2), device = DEVICE)
    halo = torch.zeros((nx, ny), device = DEVICE)
    
    
    for y, x in zip(x_sub, y_sub):
        R_sub = ((Xs-x)**2 + (Ys-y)**2)**0.5
        halo += torch.exp(-(R_sub)**2/w_sub**2/2) 
    
    
    result = {"image": halo}
    
    torch.set_default_tensor_type(torch.FloatTensor)  # HACK
    return result

# def noise(obs, _= None, sigma_n = sigma):
#     image = obs["image"]
#     eps = np.random.randn(*image.shape) * sigma_n
#     return {"image": image + eps}