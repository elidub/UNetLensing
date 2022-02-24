import torch, numpy as np
import swyft
from unet import UNET

DEVICE = 'cuda'


class CustomObservationTransform(torch.nn.Module):
    def __init__(self, observation_key: str, observation_shapes: dict):
        super().__init__()
        self.observation_key = observation_key
        self.n_features = torch.prod(torch.tensor(observation_shapes[observation_key]))
        self.online_z_score = swyft.networks.OnlineDictStandardizingLayer(observation_shapes)

    def forward(self, obs: dict) -> torch.Tensor:      
        x = self.online_z_score(obs)
        x = x[self.observation_key]
        x = x.view(len(x), self.n_features)
        return x

class CustomMarginalClassifier(torch.nn.Module):
    def __init__(self, n_marginals: int, n_features: int, nmc):
        super().__init__()
                
        self.n_marginals = n_marginals

        
        self.n_features = n_features.item() #n_features
        self.L = int(np.sqrt(self.n_features).item())
        print(nmc)
        
        self.nmc = nmc
        
        
        self.UNet = UNET(in_channels = 1, out_channels = self.nmc*2)
                
    def forward(self, sims: torch.Tensor, target_map: torch.Tensor) -> torch.Tensor:
    
        sims = sims.view(-1, self.L, self.L)
        x = self.UNet(sims)
                    
        x = x * target_map
        x = x.view(x.shape[0], self.nmc, 2, self.L, self.L)
        x = torch.sum(x, axis = 2)

        # apply sigmoid so we get between (0-1)
        # 1 - previous
        # two layers
        # mutipiply with prior to get posteriors
        # normalize posteriors
        # divide by prior to obtain ratio's again
        # return those two ratios

        return x.view(-1, self.n_features * self.nmc)

class CustomParameterTransform(torch.nn.Module):
    def __init__(self, nmc: int, L, lows, highs) -> None:
        super().__init__()
        self.nmc = nmc
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

    def forward(self, coord_v):        
         
        coord_grid = self.v_to_grid(coord_v)
        
        n_batch =  coord_grid.shape[0]
        n_coords = coord_grid.shape[1]*2/3
        assert n_coords.is_integer()

  
        z = torch.zeros((n_batch, self.nmc, self.L, self.L), device = DEVICE)
  
        if not (n_batch == 0 or n_coords == 0):
            
            x_grid, y_grid, m_grid = coord_grid.view(-1,3).T.to(DEVICE)            

            x_i = torch.floor((x_grid * self.L).flatten()).type(torch.long) 
            y_i = torch.floor((y_grid * self.L).flatten()).type(torch.long) 
            m_i = torch.floor( m_grid * self.nmc).type(torch.long) 
            b_i   = torch.floor(torch.arange(0, n_batch, 1/n_coords*2).to(DEVICE)).type(torch.long)
            
            indices = tuple(torch.stack((b_i, m_i, y_i, x_i)))
            z[indices] = 1
            

#         return z 
        return torch.cat((1-z, z), dim = 1)

    
# class CustomMarginalClassifier(torch.nn.Module):
#     def __init__(self, n_marginals: int, n_features: int, nmbins):
#         super().__init__()
                
#         self.n_marginals = n_marginals

        
#         self.n_features = n_features.item() #n_features
#         self.L = int(np.sqrt(self.n_features).item())
        
#         self.out_channels = nmbins * 2
        
#         self.UNet = UNET(in_channels = 1, out_channels = self.out_channels)
                
#     def forward(self, sims: torch.Tensor, target_map: torch.Tensor) -> torch.Tensor:
    
#         sims = sims.view(-1, self.L, self.L)
#         x = self.UNet(sims)
        
#         x = x * target_map
#         x = x.view(-1, self.n_features * self.out_channels)
        
#         # apply sigmoid so we get between (0-1)
#         # 1 - previous
#         # two layers
#         # mutipiply with prior to get posteriors
#         # normalize posteriors
#         # divide by prior to obtain ratio's again
#         # return those two ratios
        
#         return x

# class CustomParameterTransform(torch.nn.Module):
#     def __init__(self, nmbins: int, L, lows, highs) -> None:
#         super().__init__()
#         self.nmbins = nmbins
#         self.L   = L
#         self.lows = lows
#         self.highs = highs
        
#     def v_to_grid(self, coord_v):
              
#         coorv_v10 = torch.clone(coord_v)
        
#         # Transform all masses from e.g. 10^8.5 to 8.5 so they are aligned with the highs and lows
#         coorv_v10[:,2::3] = torch.log10(coorv_v10[:,2::3])
        
#         n = len(coord_v[0])/3
#         assert n.is_integer()
#         n = int(n)
          
#         lows = np.full(coord_v.shape, np.tile(self.lows, n))
#         highs = np.full(coord_v.shape, np.tile(self.highs, n))   
        
#         grid = lambda v: (v - lows) / (highs - lows)
#         coord_grid = grid(coorv_v10)
        
#         return coord_grid

#     def forward(self, coord_v):        
         
#         coord_grid = self.v_to_grid(coord_v)
        
#         n_batch =  coord_grid.shape[0]
#         n_coords = coord_grid.shape[1]*2/3
#         assert n_coords.is_integer()

  
#         z = torch.zeros((n_batch, self.nmbins, self.L, self.L), device = DEVICE)
  
#         if not (n_batch == 0 or n_coords == 0):
            
#             x_grid, y_grid, m_grid = coord_grid.view(-1,3).T.to(DEVICE)            

#             x_i = torch.floor((x_grid * self.L).flatten()).type(torch.long) 
#             y_i = torch.floor((y_grid * self.L).flatten()).type(torch.long) 
#             m_i = torch.floor( m_grid * self.nmbins).type(torch.long) 
#             b_i   = torch.floor(torch.arange(0, n_batch, 1/n_coords*2).to(DEVICE)).type(torch.long)
            
#             indices = tuple(torch.stack((b_i, m_i, y_i, x_i)))
#             z[indices] = 1
            
#         return torch.cat((1-z, z), dim = 1)