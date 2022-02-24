import os
import torch, pyro, numpy as np
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from torch import tensor
import torch.nn as nn
import torchvision.transforms.functional as TF


from clipppy import load_config, Clipppy
from clipppy.patches import torch_numpy
from ruamel.yaml import YAML

import swyft
import pyro.distributions as dist

import matplotlib.pyplot as plt
import numpy as np


SIGMA = 0.1

DEVICE = 'cuda'

def get_config(system_name: str, nsub: str = '', m: str = '') -> Clipppy:
    """
    Get configuration
    """
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # HACK

    SOURCE_DIR = '../../mock_data/sources'
        
    source_name = f'{system_name}.npy'
    config = load_config(f'configs/config_nsub{nsub}_m{m}.yaml', base_dir=SOURCE_DIR)

    torch.set_default_tensor_type(torch.FloatTensor)  # HACK
    return config


def get_prior(config: Clipppy):
    """
    Set up subhalo parameter priors using a config
    """
    main = config.umodel.alphas["main"]
    prior_p_sub = main.sub.pos_sampler.base_dist
    m_sub_grid = main.sub.mass_sampler.y
    lows = np.array(
        [
            prior_p_sub.low[0].item(),
            prior_p_sub.low[1].item(),
            m_sub_grid.min().log10().item(),
        ]
    )
    highs = np.array(
        [
            prior_p_sub.high[0].item(),
            prior_p_sub.high[1].item(),
            m_sub_grid.max().log10().item(),
        ]
    )
    
    
    nsub = main.sub.nsub
    lows_u = np.tile(lows, nsub)
    highs_u = np.tile(highs, nsub)
    
    uv = lambda u: (highs_u - lows_u) * u + lows_u
    return swyft.Prior(uv, nsub*3), uv, lows, highs


def simul(v, config: Clipppy):
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

    x_sub, y_sub, log10_m_sub = v.view(-1,3).T.to(DEVICE)
    xy_sub = torch.stack((x_sub, y_sub)).T
    
    d_m_sub = dist.Delta(10 ** log10_m_sub)
    d_p_sub = dist.Delta(xy_sub).to_event(1)


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

def noise(obs, _=None, sigma_n=SIGMA):
    image = obs["image"]
    eps = np.random.randn(*image.shape) * sigma_n
    return {"image": image + eps}

def get_sim_path(m, nsub, nsim, system_name):
    sim_name = f'_M_m{m}_nsub{nsub}_nsim{nsim}'
    sim_path = f'/nfs/scratch/eliasd/store{sim_name}.zarr' 
    print(f'Store {sim_name} exists!') if os.path.exists(sim_path) else print('Store does not exist!')
    return sim_name, sim_path

def get_post_path(sim_name, nmbins, lr, factor, patience, save_dir = 'posts'):
    save_id = f'nmbins{nmbins}_lr{np.log10(lr)}_fac{np.log10(factor)}_pat{patience}'
    save_name = f'UNet{sim_name}_{save_id}.pt'
    save_path = os.path.join(save_dir, save_name)
    return save_name, save_path

def get_losses(post):
    keys = list(post._ratios.keys())
    assert len(keys) == 1
    losses = post._ratios[keys[0]]._train_diagnostics
    assert len(losses) == 1
    tl = losses[0]['train_loss']
    vl = losses[0]['valid_loss']
    epochs = np.arange(len(tl))
    return epochs, tl, vl

def plot_losses(post, title = ''):
    fig, ax = plt.subplots(1, 1)
    
    epochs, tl, vl = get_losses(post)
        
    ax.plot(epochs, tl, '--', label = f'training loss')
    ax.plot(epochs, vl, '-', label = f'val loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    plt.legend()
    plt.show()