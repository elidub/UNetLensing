import torch, numpy as np
from pyrofit.lensing.distributions import get_default_shmf


DEVICE = 'cuda'

import torch, numpy as np
from pyrofit.lensing.distributions import get_default_shmf


class Prior():
    def __init__(self, nsub, nmc, lows, highs, L):
        self.nsub = nsub
        self.nmc  = nmc
        self.lows = lows
        self.highs = highs
        self.L = L
        
        self.m_centers, self.m_edges = self.get_m()
    
    def get_m(self):
        m = torch.logspace(self.lows[-1], self.highs[-1], 2*self.nmc+1)
        m_centers, m_edges = m[1::2], m[0::2]
        return m_centers, m_edges
    
    def __call__(self):
        
        a, b = self.lows[-1], self.highs[-1]
        shmf = get_default_shmf(z_lens = 0.5, log_range = (a, b))
        bins = torch.logspace(start = a, end = b, steps = 100)
        sample = shmf.sample((pow(10,8),)).numpy()
        print("Sampling SHMF, this should NOT happen more than once!!!")
        hist, _ = np.histogram(sample, bins = bins, density = True)
        
        m_edges_idx = torch.tensor([(bins - m_edge).abs().argmin() for m_edge in self.m_edges])
        shmf_frac = np.array([sum((torch.diff(bins)*hist)[i:j]) for i, j in zip(m_edges_idx[:-1], m_edges_idx[1:])])
        
        
        priors = self.nsub / (self.L*self.L) * shmf_frac
        prior0 = 1 - priors
        prior1 = priors
        print('Prior subhalo in Mass range')
        for m_left, m_right, prior in zip(self.m_edges[:-1], self.m_edges[1:], priors):
            print(f"{prior:.2e} \t [{m_left:.2e} - {m_right:.2e}]")

        return prior0, prior1, np.concatenate((prior0, prior1))
    

class Predict():
    def __init__(self, entry, lows, highs, L, dataset, mre):
        
        self.dataset = dataset
        self.mre = mre
       
        self.L = L
        nsub = entry['nsub']
        self.nmc = entry['nmc']
        
        prior  = Prior(nsub, self.nmc, lows, highs, L)
        _, _, self.priors = prior()
                
        self.coord_empty, self.coord_full = self.get_empty_and_full_coords(lows, highs, prior.m_centers, L)
#         _, _, self.priors = self.get_priors(nsub, lows, highs, L, self.nmc, m_edges)
                
        
    def get_empty_and_full_coords(self, lows, highs, m_centers, L):
        grid = torch.linspace(lows[0], highs[0], L+1)[:-1]
        x, y = torch.meshgrid(grid, grid, indexing = 'xy')
#         m = torch.logspace(lows[-1], highs[-1], 2*nmc+1)
#         m_centers, m_edges = m[1::2], m[0::2]
        ms = [torch.full((L*L,), m_i) for m_i in m_centers]

        coord_empty = torch.tensor((), device = DEVICE, dtype = torch.float).reshape(1, -1)
        coord_full = torch.cat( [torch.transpose(torch.stack((x.flatten(), y.flatten(), m)), 0, 1) for m in ms] ).reshape(1, -1).to(DEVICE, dtype = torch.float)

        return coord_empty, coord_full
    

    
    def get_obs(self, dataset, obs0_i = -1):
        obs0_i = np.random.randint(0, len(dataset)) if obs0_i == -1 else obs0_i

        obs0 = dataset[obs0_i][0]
        v0 = dataset[obs0_i][2]

        obs0['image'] = obs0['image'].unsqueeze(0).to(DEVICE, dtype = torch.float)
        v0 = v0.unsqueeze(0).to(DEVICE)

        return obs0, v0, obs0_i
    
    def __call__(self, obs0_i = -1):
    
        # Get observation and targets
        obs0, v0, obs0_i = self.get_obs(self.dataset, obs0_i)
        targets = self.mre.network.parameter_transform(v0).squeeze()
        target = targets[self.nmc :].numpy()
#         print(targets.shape, target.shape)
        

        # Get logratios
        logratios = np.zeros((self.nmc *2, self.L, self.L))
        logratios[:self.nmc ] = self.mre.network(obs0, self.coord_empty).view(self.nmc, self.L, self.L)
        logratios[self.nmc :] = self.mre.network(obs0, self.coord_full).view(self.nmc, self.L, self.L)
                
        
        # Posterior 
        posts = np.exp(logratios) * (self.priors)[:, np.newaxis, np.newaxis] 
        posts_sum = np.sum(posts.reshape(2, self.nmc, self.L, self.L).transpose([1,0,2,3]), axis = 1)
        posts_sum = np.tile(posts_sum, (2,1,1))
        posts_norm = posts / posts_sum
        post_norm = posts_norm[self.nmc :]

        obs0 = obs0['image'].squeeze().numpy()
        v0 = v0.numpy()

        return post_norm, target, obs0, v0, obs0_i
    
# predict = Predict(entry, lows, highs, L, dataset, mre)


# class Predict():
#     def __init__(self, entry, lows, highs, L, dataset, mre):
        
#         self.dataset = dataset
#         self.mre = mre
       
#         self.L = L
#         nsub = entry['nsub']
#         self.nmc = entry['nmc']
        
        
#         self.coord_empty, self.coord_full, m_edges = self.get_empty_and_full_coords(lows, highs, self.nmc, L)
#         _, _, self.priors = self.get_priors(nsub, lows, highs, L, self.nmc, m_edges)
                
        
#     def get_empty_and_full_coords(self, lows, highs, nmc, L):
#         grid = torch.linspace(lows[0], highs[0], L+1)[:-1]
#         x, y = torch.meshgrid(grid, grid, indexing = 'xy')
#         m = torch.logspace(lows[-1], highs[-1], 2*nmc+1)
#         m_centers, m_edges = m[1::2], m[0::2]
#         ms = [torch.full((L*L,), m_i) for m_i in m_centers]

#         coord_empty = torch.tensor((), device = DEVICE, dtype = torch.float).reshape(1, -1)
#         coord_full = torch.cat( [torch.transpose(torch.stack((x.flatten(), y.flatten(), m)), 0, 1) for m in ms] ).reshape(1, -1).to(DEVICE, dtype = torch.float)

#         return coord_empty, coord_full, m_edges
    
#     def get_priors(self, nsub, lows, highs, L, nmc, m_edges):
        
#         a, b = lows[-1], highs[-1]
#         shmf = get_default_shmf(z_lens = 0.5, log_range = (a, b))
#         bins = torch.logspace(start = a, end = b, steps = 100)
#         sample = shmf.sample((pow(10,8),)).numpy()
#         hist, _ = np.histogram(sample, bins = bins, density = True)
        
#         m_edges_idx = torch.tensor([(bins - m_edge).abs().argmin() for m_edge in m_edges])
#         shmf_frac = np.array([sum((torch.diff(bins)*hist)[i:j]) for i, j in zip(m_edges_idx[:-1], m_edges_idx[1:])])
        
        
#         priors = nsub / (L*L) * shmf_frac
#         prior0 = 1 - priors
#         prior1 = priors
#         print('Prior subhalo in Mass range')
#         for m_left, m_right, prior in zip(m_edges[:-1], m_edges[1:], priors):
#             print(f"{prior:.2e} \t [{m_left:.2e} - {m_right:.2e}]")

#         return prior0, prior1, np.concatenate((prior0, prior1))#, np.repeat(np.array([prior0, prior1]), nmc) 
    
#     def get_obs(self, dataset, obs0_i = -1):
#         obs0_i = np.random.randint(0, len(dataset)) if obs0_i == -1 else obs0_i

#         obs0 = dataset[obs0_i][0]
#         v0 = dataset[obs0_i][2]

#         obs0['image'] = obs0['image'].unsqueeze(0).to(DEVICE, dtype = torch.float)
#         v0 = v0.unsqueeze(0).to(DEVICE)

#         return obs0, v0, obs0_i
    
#     def __call__(self, obs0_i = -1):
    
#         # Get observation and targets
#         obs0, v0, obs0_i = self.get_obs(self.dataset, obs0_i)
#         targets = self.mre.network.parameter_transform(v0).squeeze()
#         target = targets[self.nmc :].numpy()
# #         print(targets.shape, target.shape)
        

#         # Get logratios
#         logratios = np.zeros((self.nmc *2, self.L, self.L))
#         logratios[:self.nmc ] = self.mre.network(obs0, self.coord_empty).view(self.nmc, self.L, self.L)
#         logratios[self.nmc :] = self.mre.network(obs0, self.coord_full).view(self.nmc, self.L, self.L)
                
        
#         # Posterior 
#         posts = np.exp(logratios) * (self.priors)[:, np.newaxis, np.newaxis] 
#         posts_sum = np.sum(posts.reshape(2, self.nmc, self.L, self.L).transpose([1,0,2,3]), axis = 1)
#         posts_sum = np.tile(posts_sum, (2,1,1))
#         posts_norm = posts / posts_sum
#         post_norm = posts_norm[self.nmc :]

#         obs0 = obs0['image'].squeeze().numpy()
#         v0 = v0.numpy()

#         return post_norm, target, obs0, v0, obs0_i