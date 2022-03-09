import pylab as plt
import numpy as np
import torch
import torch.nn as nn
import pyro
import typing as tp
from torch.nn import functional as F
from fft_conv_pytorch import FFTConv2d

from pyrofit.utils import pad_dims
from pyrofit.lensing.utils import get_meshgrid
from pyrofit.lensing.lenses import SPLELens, SPLEwithHaloes, KeOpsNFWSubhaloes
from pyrofit.lensing.sources import AnalyticSource, SersicSource
from pyrofit.utils.torchutils import unravel_index
from pyrofit.utils import kNN

from pyrofit.lensing.distributions import get_default_shmf
from pyro import distributions as dist

NPIX_SRC = NPIX_IMG = 40  # Some bugs
CHANNELS = [0.5, 1, 2, 4, 8, 16, 32]
DEVICE = 'cuda'


class RandomSource:
    def __init__(self, npix =  NPIX_SRC, channels = CHANNELS, K = 129):
        self._K = K
        self._channels = channels
        self._npix = npix
        
        self._kernels = self._get_kernels(self._K, self._channels)
        self._conv = self._get_kernel_conv(self._kernels)
        
    @staticmethod
    def _get_kernels(K, channels):
        C = len(channels)
        kernel = np.zeros((C, C, K, K))
        x = np.linspace(-64, 64, K)
        X, Y = np.meshgrid(x, x)
        R = (X**2 + Y**2)**0.5

        for i, s in enumerate(channels):
            kern = np.exp(-0.5*R**2/s**2)
            kern /= (kern**2).sum()**0.5
            kernel[i, i] = kern

        return kernel

    @staticmethod
    def _get_kernel_conv(kernel):
        C = len(kernel)
        K = len(kernel[0][0])
        gaussian_weights = nn.Parameter(torch.tensor(kernel).float().cuda())
        conv = FFTConv2d(in_channels = C, out_channels = C, kernel_size=K, bias=False, padding = int(K/2))
        with torch.no_grad():
            conv.weight = gaussian_weights
        return conv

    def _get_source_image(self, seeds, A=0.5, B = 0.3, C = 0.55, D = 4.):
        scales = A*(np.array(self._channels)/max(self._channels))**B
        seeds = seeds * torch.tensor(scales).cuda().float().unsqueeze(1).unsqueeze(2)
        x = torch.linspace(-1, 1, self._npix).to(seeds.device)
        X, Y = torch.meshgrid([x, x])
        R = (X**2 + Y**2)**0.5
        imgc = self._conv(seeds.unsqueeze(0)).squeeze(0).squeeze(0)
        img = imgc.sum(axis=-3)
        img = torch.exp(img)*(1/(1+(R/C)**D))
        return img

    def __call__(self):
        C = len(self._channels)
        return self._get_source_image(torch.randn(C, self._npix, self._npix).cuda()).detach()
    
    
class ArraySource(AnalyticSource):
    def __init__(self, image_array, x: float = 0., y: float = 0., scale: float = 1., peak_intensity: float = 1.,
                 origin: tp.Literal['lower', 'upper'] = 'lower', aspect=None, device=None):
        super().__init__(device=device)

        self.x, self.y = x, y
        self.peak_intensity = peak_intensity

        self.image = self._image(image_array, peak_intensity, device)

        if aspect is None:
            aspect = self.image.shape[-2] / self.image.shape[-1]
        self.semi_scale = torch.tensor([scale, (-1 if origin == 'lower' else 1) * aspect * scale], device=device) / 2

    def _image(self, image_array, peak_intensity=None, device=None) -> torch.Tensor:
        image = torch.tensor(image_array)
        if image.shape[-1] in (1, 3, 4):
            image = torch.movedim(image, -1, -3)
        if peak_intensity is not None:
            image = image.to(torch.get_default_dtype())
            image = image / torch.amax(image, (-2, -1), keepdim=True) * peak_intensity
        self.rollback_dims = image.ndim
        return pad_dims(image, ndim=4)[0].to(device=device, dtype=torch.get_default_dtype())

    def flux(self, X, Y):
        grid = torch.stack((X - self.x, Y - self.y), dim=-1).reshape(-1, *X.shape[-2:], 2) / self.semi_scale

        return F.grid_sample(
            self.image.expand(grid.shape[0], *self.image.shape[-3:]),
            grid,
            align_corners=True
        ).reshape(*X.shape[:-2], *self.image.shape[-self.rollback_dims:-2], *X.shape[-2:])
    
    
def get_kNN_idx(X, Y, Xsrc, Ysrc, k = 1):
        """Return indices into `Xsrc` and `Ysrc` closest to each point in `psrc`."""
        P = torch.stack((X, Y), -1).flatten(-3, -2)
        Psrc = torch.stack((Xsrc, Ysrc), -1).flatten(-3, -2)
        idx = unravel_index(kNN(P, Psrc, k).squeeze(-1), Xsrc.shape[-2:])
        #print(X.shape, Y.shape, Xsrc.shape, Ysrc.shape, NPIX, idx.shape)
        idx = torch.reshape(idx, (NPIX_IMG, NPIX_IMG, k, 2))
        return idx
    
    
def deproject_idx(image, kNN_idx):
        """Return indices into `Xsrc` and `Ysrc` closest to each point in `psrc`."""
        k = kNN_idx.shape[-2]
        B = image.shape[0]
        #print(k, B)
        #print(image.shape)
        #print(kNN_idx.shape)
        # TODO: Need to speed up nested python loops
        src_image = torch.stack([torch.stack([image[b, kNN_idx[b, ..., i,0], kNN_idx[b, ..., i,1]] for i in range(k)]) for b in range(B)])
        return src_image
    
    
def image_generator(x, y, phi, q, r_ein, slope, src_image):
    res = 0.0125*8*50/NPIX_IMG       # resolution in arcsec
    nx, ny = NPIX_IMG, NPIX_IMG                   # number of pixels
    X, Y = get_meshgrid(res, nx, ny)    # grid

    # Lens
    lens = SPLELens(device='cuda')
    lens.XY = X,Y
    # Displacement field
    alphas = lens(x=x, y=y, phi=phi, q=q, r_ein=r_ein, slope=slope)

    # Lensing equation
    X_src = X - alphas[..., 0, :, :]
    Y_src = Y - alphas[..., 1, :, :]

    # Source
    source = ArraySource(src_image, peak_intensity = None, scale = 1)

    # Lensed source
    image = source(X=X_src, Y=Y_src)
    
    return image, [X, Y, X_src, Y_src]


def image_generator_sersic(x, y, phi, q, r_ein, slope, x_src, y_src, phi_src, q_src, index, r_e, I_e):
    res = 0.0125*8*50/NPIX_IMG       # resolution in arcsec
    nx, ny = NPIX_IMG, NPIX_IMG                   # number of pixels
    X, Y = get_meshgrid(res, nx, ny)    # grid

    # Lens
    lens = SPLELens(device='cuda')
    lens.XY = X,Y
    # Displacement field
    alphas = lens(x=x, y=y, phi=phi, q=q, r_ein=r_ein, slope=slope)

    # Lensing equation
    X_src = X - alphas[..., 0, :, :]
    Y_src = Y - alphas[..., 1, :, :]

    # Source
    sersic = SersicSource()

    # Lensed source
    image = sersic(X=X_src, Y=Y_src, x=x_src, y=y_src, phi=phi_src, q=q_src, index=index, r_e=r_e, I_e=I_e)
    
    return image, [X, Y, X_src, Y_src]

def image_generator_sub(x, y, phi, q, r_ein, slope, x_src, y_src, phi_src, q_src, index, r_e, I_e, x_sub, y_sub, M_sub):
    res = 0.0125*8*50/NPIX_IMG       # resolution in arcsec
    nx, ny = NPIX_IMG, NPIX_IMG                   # number of pixels
    X, Y = get_meshgrid(res, nx, ny)    # grid
    
    z_lens, z_src = 0.5, 2.
    c_200c = 15.
    
    z_lens, z_src = torch.tensor((0.5), device='cuda'), torch.tensor((2.), device='cuda')
    nsub = 1
#     pos_sampler  = dist.Uniform(torch.tensor(-2.5), torch.tensor(2.5)).expand([2]).to_event(1)
#     mass_sampler = get_default_shmf(z_lens=z_lens, log_range=[10., 11.])
    
    xy_sub = torch.stack((x_sub, y_sub)).T
    
    mass_sampler = d_m_sub = pyro.distributions.Delta(M_sub)
    pos_sampler = d_p_sub = pyro.distributions.Delta(xy_sub).to_event(1)
    
    c_200c_sampler = 15.
    
    

    
    
    sub = KeOpsNFWSubhaloes(z_lens=z_lens, z_src=z_src,
                            nsub=nsub,pos_sampler=pos_sampler, mass_sampler=mass_sampler, c_200c_sampler=c_200c_sampler,
                            device='cuda')

    # Lens
    lens = SPLEwithHaloes(z_lens=z_lens, z_src=z_src, 
                          sub=sub,
                          device='cuda')
    lens.XY = X,Y
    # Displacement field
    alphas = lens(
                  x=x, y=y, phi=phi, q=q, r_ein=r_ein, slope=slope)

    # Lensing equation
    X_src = X - alphas[..., 0, :, :]
    Y_src = Y - alphas[..., 1, :, :]

    # Source
    sersic = SersicSource()

    # Lensed source
    image = sersic(X=X_src, Y=Y_src, x=x_src, y=y_src, phi=phi_src, q=q_src, index=index, r_e=r_e, I_e=I_e)
    
    return image, [X, Y, X_src, Y_src]

def image_generator_toy(x, y, phi, q, r_ein, slope, x_src, y_src, phi_src, q_src, index, r_e, I_e, z_sub):
    res = 0.0125*8*50/NPIX_IMG       # resolution in arcsec
    nx, ny = NPIX_IMG, NPIX_IMG                   # number of pixels
    X, Y = get_meshgrid(res, nx, ny)    # grid
    
    
    
#     for x, y in zip(x_sub, y_sub):
    w_sub = torch.tensor((0.2), device = DEVICE)
    blobs = torch.zeros((nx, ny), device = DEVICE)
    
    for x_sub, y_sub, M_sub in z_sub:
        R_sub = ((X-x_sub)**2 + (Y-y_sub)**2)**0.5
        blobs += torch.exp(-(R_sub)**2/(M_sub/1e11)**2/2) 
    
    image = blobs
    
    X_src, Y_src = X, Y

    
    return image, [X, Y, X_src, Y_src]