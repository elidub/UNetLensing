import torch, numpy as np
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from torch import tensor
import torch.nn as nn
import torchvision.transforms.functional as TF

import swyft

DEVICE = 'cuda'

class Mapping:
    def __init__(self, nmbins, L, lows, highs):
        self.nmbins = nmbins
        self.L   = L
        self.lows = lows
        self.highs = highs

    def coord_vu(self, coords_v):
                        
        n = len(coords_v[0])/3
        assert n.is_integer()
        n = int(n)

        lows = np.full(coords_v.shape, np.tile(self.lows, n))
        highs = np.full(coords_v.shape, np.tile(self.highs, n))   
                
        u = lambda v: (v - lows) / (highs - lows)
        coords_u = u(coords_v)
        return coords_u

    def coord_to_map(self, XY_u):

        
        n_batch =  XY_u.shape[0]
        n_coords = XY_u.shape[1]*2/3
        assert n_coords.is_integer()

        z = torch.zeros((n_batch, self.nmbins + 1, self.L, self.L), device = DEVICE)
                
        if not (n_batch == 0 or n_coords == 0):
            
            x_sub_u, y_sub_u, log10_m_sub_u = XY_u.view(-1,3).T.to(DEVICE)

            x_i = torch.floor((x_sub_u*self.L).flatten()).type(torch.long) 
            y_i = torch.floor((y_sub_u*self.L).flatten()).type(torch.long) 
            m_i = torch.floor( log10_m_sub_u * self.nmbins ).type(torch.long) + 1
            
            i   = torch.floor(torch.arange(0, n_batch, 1/n_coords*2).to(DEVICE)).type(torch.long)
            xx = tuple(torch.stack((i, m_i, y_i, x_i)))
            z[xx] = 1

            xx = tuple(torch.stack((i, torch.zeros_like(m_i), y_i, x_i)))
            z[xx] = 1
            
        z[:,0] = 1 - z[:,0]

        return z


class DoubleConv(swyft.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), # bias = False becaise BatchNorm2d is set
            nn.BatchNorm2d(out_channels), # BatchNorm2d were not known when paper came out
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(swyft.Module):
    def __init__(self, in_channels, out_channels, features = [64, 128, 256, 512]):
        super(UNET, self).__init__()
                
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # keep size the same
        

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        

    def forward(self, x, target):
                
        x = x.unsqueeze(1)
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # reverse list

        # the upsampling
        for idx in range(0, len(self.ups), 2): # step of 2 because we want up - double column - up - double column
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2] # //2 because we want still steps of one

            # if statement because we can put in shapes that are not divisble by two around 19:00 of video
            if x.shape != skip_connection.shape: 
                x = TF.resize(x, size=skip_connection.shape[2:]) # hopefully does not impact accuracy too much

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        x = self.final_conv(x)
        return x

    
class CustomHead(swyft.Module):
    def __init__(self, obs_shapes) -> None:
        super().__init__(obs_shapes=obs_shapes)
        self.n_features = torch.prod(tensor(obs_shapes['image']))
#         self.onl_norm = OnlineNormalizationLayer(torch.Size([self.n_features]))

    def forward(self, obs) -> torch.Tensor:
        x = obs["image"]
        n_batch = len(x)
        x = x.view(n_batch, self.n_features)
#         x = self.onl_norm(x)    
        return x


class CustomTail(swyft.Module):
    def __init__(self, n_features, marginals, **tail_args):
        super().__init__(n_features = n_features, marginals = marginals, **tail_args)
        
        
        self.n_features = n_features
        self.L = int(np.sqrt(n_features).item())
        self.nmbins = tail_args['nmbins']
        self.lows   = tail_args['lows']
        self.highs  = tail_args['highs']
        self.out_channels = self.nmbins + 1
        
        self.Map  = Mapping(self.nmbins, self.L, self.lows, self.highs)
        self.UNet = UNET(in_channels = 1, out_channels = self.out_channels)
        
       
    def forward(self, sims, target):
        
        sims = sims.view(-1, self.L, self.L)
        
        x = self.UNet(sims, target)
        z = self.Map.coord_to_map(target)
        
        x = x * z
        x = x.view(-1, self.n_features * self.out_channels)
        
        return x
