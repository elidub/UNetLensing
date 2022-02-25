import torch, numpy as np
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from torch import tensor
import torch.nn as nn
import torchvision.transforms.functional as TF

import swyft

DEVICE = 'cuda'
    

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
    def __init__(self, n_features, marginals):
        super().__init__(n_features, marginals) 
#         super(UNET, self).__init__()
        
        self.marginals = marginals
        self.n_features = n_features
        
        self.L = 40
        
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # keep size the same
        
        in_channels=1
        out_channels=2
        features=[64, 128, 256, 512]

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
        
    def coord_uv(self, coords_u, lows, highs):
    #     highs_l = np.repeat(highs, coords_u)
    #     lows_l = np.repeat(lows, coords_u)
        highs_l = np.full_like(coords_u, highs)
        lows_l = np.full_like(coords_u, lows)

        v = lambda u: (highs_l - lows_l) * u + lows_l
        coords_v = v(coords_u)
        return coords_v

    def coord_to_map(self, XY_u):

        y0, y1, x0, x1 = -2.5, 2.5, -2.5, 2.5
        lows, highs = -2.5, 2.5
        res = 0.125

        XY = XY_u

        n_batch =  XY.shape[0]
        n_coords = XY.shape[1]

        binary_map = torch.zeros((n_batch, self.L,self.L), device = DEVICE)

        x, y = XY[:,0::2], XY[:,1::2]

        x_i = torch.floor((x*self.L).flatten()).type(torch.long) 
        y_i = torch.floor((y*self.L).flatten()).type(torch.long) 

        if n_coords != 0:
            i   = torch.floor(torch.arange(0, n_batch, 1/n_coords*2).to(DEVICE)).type(torch.long) 

            xx = tuple(torch.stack((i, y_i, x_i)))
            binary_map[xx] = 1

        return binary_map
        

    def forward(self, sims, target):
                
        sims = sims.view(-1, self.L, self.L)
        z = self.coord_to_map(target)
    
        ############# UNet Start ###
        x = sims
        n_batch = len(x)
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
        ############# UNet End ###

        
                
        # L[C]
        x_new = x[:,0] * (1 - z) + x[:,1] * z
        
        
#         if len(x) != 0:
            
#             u = target[0]
#             v = self.coord_uv(target[0].numpy(), -1, 1)

        
#             plt_imshow([x[0][0], x[0][1], x_new[0], z[0], z2[0]], 
#                        titles = ['lnr0', 'lnr1', 'lnr[z]', 'z', 'z2'],
# #                        scatter = [v],
#                        cbar = True, size_y = 3, **imkwargs)
        
        x = x_new
        x = x.view(-1, self.n_features)
        return x

class CustomHead(swyft.Module):

    def __init__(self, obs_shapes) -> None:
        super().__init__(obs_shapes=obs_shapes)
        self.n_features = torch.prod(tensor(obs_shapes['image']))

    def forward(self, obs) -> torch.Tensor:
        x = obs["image"]
        n_batch = len(x)
        x = x.view(n_batch, self.n_features)
        return x