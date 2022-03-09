import torch, numpy as np
from pyrofit.lensing.distributions import get_default_shmf

from plot import *



DEVICE = 'cuda'


class Prior():
    def __init__(self, nsub, nmc, L, low, high):
        self.nsub = nsub
        self.nmc  = nmc
        self.L = L
        self.low = low
        self.high = high
        
        
        self.m_centers, self.m_edges = self.get_m()
#         self.prior = self.calc_prior()
    
    def get_m(self):
        ic(self.low[-1], self.high[-1])
        m = torch.logspace(self.low[-1], self.high[-1], 2*self.nmc+1)
        ic( m )
        m_centers, m_edges = m[1::2], m[0::2]
        ic(m_centers, m_edges)
        return m_centers, m_edges
    
    def calc_prior(self):
        a, b = self.low[-1], self.high[-1]
        ic(a, b)
        shmf = get_default_shmf(z_lens = 0.5, log_range = (a, b))
        bins = torch.logspace(start = a, end = b, steps = 100)
        sample = shmf.sample((pow(10,6),)).numpy()
#         print("Sampling SHMF, this should NOT happen more than once!!!")
        hist, _ = np.histogram(sample, bins = bins, density = True)
        
        m_edges_idx = torch.tensor([(bins - m_edge).abs().argmin() for m_edge in self.m_edges])
        shmf_frac = torch.tensor([sum((torch.diff(bins)*hist)[i:j]) for i, j in zip(m_edges_idx[:-1], m_edges_idx[1:])]).to(DEVICE)
        
        ic( shmf_frac, torch.sum(shmf_frac) )
        ic(self.nsub, self.L)
        ic(self.m_edges)
        
        
        prior = self.nsub / (self.L*self.L) * shmf_frac
        
        print('Prior subhalo in Mass range')
        for m_left, m_right, p in zip(self.m_edges[:-1], self.m_edges[1:], prior):
            print(f"{p:.2e} \t [{m_left:.2e} - {m_right:.8e}]")
            
        return prior
        
    
    def calc_priors(self, prior):
        
        prior = prior.unsqueeze(1).unsqueeze(1).repeat(1, self.L, self.L)

        return torch.cat((1-prior, prior), dim = 0)

class Predict():
    def __init__(self, nsub, nmc, L, low, high):
        self.nsub = nsub
        self.nmc  = nmc
        self.L = L
        self.low = low
        self.high = high
        
       
        prior  = Prior(self.nsub, self.nmc, L, low, high)
        self.m_centers, self.m_edges = prior.get_m()
        self.prior  = prior.calc_prior()
        self.priors = prior.calc_priors(self.prior)
                
        self.z_sub_empty, self.z_sub_all = self.get_1_z_sub(low, high, prior.m_edges, L)
#         _, _, self.priors = self.get_priors(nsub, low, high, L, self.nmc, m_edges)
                
        
    def get_1_z_sub(self, low, high, m_edges, L):
        grid = torch.linspace(low[0], high[0], L+1)[:-1]
        x, y = torch.meshgrid(grid, grid, indexing = 'xy')
        ms = [torch.full((x.shape), m_i) for m_i in m_edges[:-1]]

        z_sub_empty = torch.tensor((), device = DEVICE, dtype = torch.float).reshape(1, 0, 3).to(DEVICE)
        z_sub_all   = torch.cat([ torch.stack((x, y, m), dim = 2).flatten(end_dim = 1) for m in ms]).unsqueeze(0).to(DEVICE)

        return dict(z_sub = z_sub_empty), dict(z_sub = z_sub_all)
    

    
    def get_obs(self, s0):
        
        s0_copy = s0.copy()
        for k, v in s0_copy.items():
            s0_copy[k] = v.unsqueeze(0)

        return s0_copy
    
    def __call__(self, r1, s0):
    
        # Get observation and targets
#         s0 = self.get_obs(s0)

        N = nbatch = len(s0['img'])

        # Get logratios
        N_z_sub_empty = self.z_sub_empty.copy()
        N_z_sub_empty['z_sub'] = N_z_sub_empty['z_sub'].repeat(N, 1, 1)
        
        N_z_sub_all = self.z_sub_all.copy()
        N_z_sub_all['z_sub'] = N_z_sub_all['z_sub'].repeat(N, 1, 1)
        
        logratios = torch.zeros((nbatch, self.nmc*2, self.L, self.L), device = DEVICE)
        logratios[:,:self.nmc] = r1(s0, N_z_sub_empty)['z_pix'].ratios
        logratios[:,self.nmc:] = r1(s0, N_z_sub_all)['z_pix'].ratios
                
        
        # Posterior   
        posts = torch.exp(logratios) * self.priors
        post_sum = torch.sum(torch.transpose( posts.reshape(nbatch, 2, self.nmc, self.L, self.L), 1, 2), dim = 2)
        post_norm = posts[:,self.nmc:] / post_sum

        return post_norm