import torch, numpy as np
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from torch import tensor
import torch.nn as nn
import torchvision.transforms.functional as TF

import swyft
from swyft.networks.standardization import OnlineStandardizingLayer
from swyft.types import MarginalIndex

DEVICE = 'cuda'

class Mapping:
    def __init__(self, nmbins, L, lows, highs):
        self.nmbins = nmbins
        self.L   = L
        self.lows = lows
        self.highs = highs
        
    def v_to_grid(self, coord_v):
              
        coorv_v10 = torch.clone(coord_v)
        
        # Transform all masses from e.g. 10^8.5 to 8.5 so they are aligned with the highs and lows
        coorv_v10[:,2::3] = torch.log10(coorv_v10[:,2::3])
        
        n = len(coord_v[0])/3
        assert n.is_integer()
        n = int(n)
          
        lows = np.full(coord_v.shape, np.tile(self.lows, n))
        highs = np.full(coord_v.shape, np.tile(self.highs, n))   
        
        grid = lambda v: (v - lows) / (highs - lows)
        coord_grid = grid(coorv_v10)
        
        
        return coord_grid

    def coord_to_map(self, coord_v):        
         
        coord_grid = self.v_to_grid(coord_v)
        
        n_batch =  coord_grid.shape[0]
        n_coords = coord_grid.shape[1]*2/3
        assert n_coords.is_integer()

        
        z0 = torch.ones((n_batch, self.nmbins, self.L, self.L), device = DEVICE)
        z1 = torch.zeros((n_batch, self.nmbins, self.L, self.L), device = DEVICE)

                
        if not (n_batch == 0 or n_coords == 0):
            
            x_grid, y_grid, m_grid = coord_grid.view(-1,3).T.to(DEVICE)            

            x_i = torch.floor((x_grid * self.L).flatten()).type(torch.long) 
            y_i = torch.floor((y_grid * self.L).flatten()).type(torch.long) 
            m_i = torch.floor( m_grid * self.nmbins).type(torch.long) 
            b_i   = torch.floor(torch.arange(0, n_batch, 1/n_coords*2).to(DEVICE)).type(torch.long)
            
            indices = tuple(torch.stack((b_i, m_i, y_i, x_i)))
            z0[indices], z1[indices] = 0, 1
            
        return torch.cat((z0, z1), dim = 1), z1



class DoubleConv(nn.Module):
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

class UNET(nn.Module):
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
        

    def forward(self, x):
                
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

class CustomObservationTransform(torch.nn.Module):
    def __init__(self, observation_key: str, observation_shapes: dict):
        super().__init__()
        self.observation_key = observation_key
        self.n_features = torch.prod(tensor(observation_shapes[observation_key]))

    def forward(self, obs: dict) -> torch.Tensor:      
        x = obs
        x = x[self.observation_key]
        x = x.view(len(x), self.n_features)
        return x

class CustomMarginalClassifier(torch.nn.Module):
    def __init__(self, n_marginals: int, n_features: int, args):
        super().__init__()
                
        self.n_marginals = n_marginals

        
        self.n_features = n_features.item() #n_features
        self.L = int(np.sqrt(self.n_features).item())
        self.nmbins = args['nmbins']
        self.lows   = args['lows']
        self.highs  = args['highs']
        
        self.out_channels = self.nmbins
        
        self.Map  = Mapping(self.nmbins, self.L, self.lows, self.highs)
        self.UNet = UNET(in_channels = 1, out_channels = self.out_channels)
                
    def forward(
        self, features: torch.Tensor, marginal_block: torch.Tensor
    ) -> torch.Tensor:
        sims = features
        target = marginal_block
    
        sims = sims.view(-1, self.L, self.L)
        x = self.UNet(sims)
        z_both, z = self.Map.coord_to_map(target)
        
        x = x * z
        x = x.view(-1, self.n_features * self.out_channels)
        
        return x
    
class CustomParameterTransform(nn.Module):
    def __init__(
        self, n_parameters: int, marginal_indices: MarginalIndex, online_z_score: bool
    ) -> None:
        super().__init__()
        self.n_parameters = torch.Size([n_parameters])
        if online_z_score:
            self.online_z_score = OnlineStandardizingLayer(self.n_parameters)
        else:
            self.online_z_score = nn.Identity()

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        return self.online_z_score(parameters)