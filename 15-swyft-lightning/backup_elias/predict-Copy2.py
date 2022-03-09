import torch, numpy as np
from pyrofit.lensing.distributions import get_default_shmf
from tqdm import tqdm

from plot import *



DEVICE = 'cuda'


class Prior():
    def __init__(self, n_sub, n_pix, n_msc, grid_low, grid_high):
        self.n_sub = n_sub
        self.n_pix = n_pix
        self.n_msc = n_msc
        self.grid_low = grid_low 
        self.grid_high = grid_high
        
        
        self.m_centers, self.m_edges = self.get_m()
#         self.prior = self.calc_prior()
    
    def get_m(self):
        m = torch.logspace(self.grid_low[0], self.grid_high[0], 2*self.n_msc+1)
        m = torch.log10(m)
        m_centers, m_edges = m[1::2], m[0::2]
        return m_centers, m_edges
    
    def calc_prior(self):
        a, b = self.grid_low[0], self.grid_high[0]
        
#         ic(a, b)
#         shmf = get_default_shmf(z_lens = 0.5, log_range = (a, b))
#         bins = torch.logspace(start = a, end = b, steps = 100)
#         sample = shmf.sample((pow(10,6),)).numpy()
# #         print("Sampling SHMF, this should NOT happen more than once!!!")
#         hist, _ = np.histogram(sample, bins = bins, density = True)
        
#         m_edges_idx = torch.tensor([(bins - m_edge).abs().argmin() for m_edge in self.m_edges])
#         shmf_frac = torch.tensor([sum((torch.diff(bins)*hist)[i:j]) for i, j in zip(m_edges_idx[:-1], m_edges_idx[1:])]).to(DEVICE)
        
        shmf_frac = torch.full((self.n_msc,), 1/self.n_msc, device = DEVICE)
        
        
        prior = self.n_sub / (self.n_pix*self.n_pix) * shmf_frac
        
        print('Prior subhalo in Mass range')
        for m_left, m_right, p in zip(self.m_edges[:-1], self.m_edges[1:], prior):
            print(f"{p:.2e} \t [{m_left:.2e} - {m_right:.8e}]")
            
        return prior
        
    
    def calc_priors(self, prior):
        
        prior = prior.unsqueeze(1).unsqueeze(1).repeat(1, self.n_pix, self.n_pix)

        return torch.cat((1-prior, prior), dim = 0)

class Infer():
    def __init__(self, n_sub, n_pix, n_msc, grid_low, grid_high):
        self.n_sub = n_sub
        self.n_pix = n_pix
        self.n_msc = n_msc
        self.grid_low = grid_low 
        self.grid_high = grid_high
        
        
       
        prior  = Prior(n_sub, n_pix, n_msc, grid_low, grid_high)
        self.m_centers, self.m_edges = prior.get_m()
        self.prior  = prior.calc_prior()
        self.priors = prior.calc_priors(self.prior)
                
        self.z_sub_empty, self.z_sub_all = self.get_1_z_sub(grid_low, grid_high, prior.m_edges, n_pix)
                
        
    def get_1_z_sub(self, grid_low, grid_high, m_edges, n_pix):
        grid = torch.linspace(grid_low[1], grid_high[1], n_pix+1)[:-1]
        x, y = torch.meshgrid(grid, grid, indexing = 'xy')
        ms = [torch.full((x.shape), m_i) for m_i in m_edges[:-1]]

        z_sub_empty = torch.tensor((), device = DEVICE, dtype = torch.float).reshape(1, 0, 3).to(DEVICE)
        z_sub_all   = torch.cat([ torch.stack((m, x, y), dim = 2).flatten(end_dim = 1) for m in ms]).unsqueeze(0).to(DEVICE)

        return dict(z_sub = z_sub_empty), dict(z_sub = z_sub_all)
    

    
    def get_obs(self, s0):
        
        s0_copy = s0.copy()
        for k, v in s0_copy.items():
            s0_copy[k] = v.unsqueeze(0)

        return s0_copy
    
    def predict(self, net, s0):
#         for k, v in s0.items():
#             print(k, v.shape)
        
    
        # Get observation and targets
#         s0 = self.get_obs(s0)

        N = nbatch = len(s0['img'])

        # Get logratios
        N_z_sub_empty = self.z_sub_empty.copy()
        N_z_sub_empty['z_sub'] = N_z_sub_empty['z_sub'].repeat(N, 1, 1)
        
        N_z_sub_all = self.z_sub_all.copy()
        N_z_sub_all['z_sub'] = N_z_sub_all['z_sub'].repeat(N, 1, 1)
        
        logratios = torch.zeros((nbatch, self.n_msc*2, self.n_pix, self.n_pix), device = DEVICE)
        logratios[:,:self.n_msc] = net(s0, N_z_sub_empty)['z_pix'].ratios
        logratios[:,self.n_msc:] = net(s0, N_z_sub_all)['z_pix'].ratios
                
        
        # Posterior   
        posts = torch.exp(logratios) * self.priors
        post_sum = torch.sum(torch.transpose( posts.reshape(nbatch, 2, self.n_msc, self.n_pix, self.n_pix), 1, 2), dim = 2)
        post_norm = posts[:,self.n_msc:] / post_sum

        return post_norm.detach()
    
    def sample(self, net, dataset, max_n_test):
        n_store = len(dataset.store['img'])
        max_n_test = max_n_test if max_n_test <= n_store else n_store
        max_n_test_batch = max_n_test // dataset.batch_size + 1

        posts, targets = [], []
        for _, s_batch in tqdm(zip(range(max_n_test_batch), dataset.test_dataloader()), total = max_n_test_batch):
            posts.append(  self.predict(net, s_batch[0]) )
            targets.append( net.classifier.paramtrans(s_batch[0]['z_sub'].to(DEVICE)) )
        posts = torch.cat(posts)
        targets = torch.cat(targets)
        return posts, targets
        
    
class LogData():
    def __init__(self, posts, targets, n_alpha):
#         super.__init__()
#         self.net = net
#         self.dataset = datasetn
        
#         self.n_sub = n_sub
#         self.n_pix = n_pix
#         self.n_msc = n_msc
#         self.grid_low = grid_low 
#         self.grid_high = grid_high
        
#         self.test_dataloader = dataset.test_dataloader()
        
#         self.predict = Predict(  n_sub, n_pix, n_msc, grid_low, grid_high)
#         self.posts, self.targets = self.get_predictions(max_n_test = max_n_test)
        self.posts = posts
        self.targets = targets
        self.n_alpha = n_alpha


        self.alpha_edges, self.alpha_centers = self.get_alpha(n_alpha = self.n_alpha)
        
#     def get_predictions(self, max_n_test):
        
        
    
    def get_alpha(self, n_alpha):
        alpha_edges = torch.linspace(0, 1, n_alpha, device = DEVICE)#, dtype=torch.float64)
        alpha_centers = (alpha_edges[:-1] + alpha_edges[1:])/2
        return alpha_edges, alpha_centers
    
    def get_histogram(self):
        hist = torch.histogram(self.posts.flatten().cpu(), bins = self.alpha_edges.cpu())[0].to(DEVICE)
        return hist
    
    def get_relicurve(self, n_alpha):

        posts_alpha = torch.repeat_interleave(self.posts.unsqueeze(-1), n_alpha-1, dim = -1)
        targets_alpha = torch.repeat_interleave(self.targets.unsqueeze(-1), n_alpha-1, dim = -1)
        
        is_between = (posts_alpha > self.alpha_edges[:-1]) & (posts_alpha < self.alpha_edges[1:])
        is_between_sum = torch.sum(targets_alpha * is_between, dim = (0, 1, 2, 3))
        hist = self.get_histogram() 
        relicurve = is_between_sum/hist
        
        return relicurve

    
    def get_sum_posts(self):
        assert len(self.posts.shape) == 4
#         assert list(self.posts.shape)[1:] == [self.n_msc, self.n_pix, self.n_pix]
        return torch.sum(self.posts, axis = (1, 2, 3)).cpu()
    
    
class LogPlots(LogData):
    def __init__(self, tbl, posts, targets, n_alpha = 50, dpi = 250):
        super().__init__(posts, targets, n_alpha)
        self.tbl = tbl
        self.dpi = dpi

    
    def plot_relicurve(self):
        
        relicurve = self.get_relicurve(self.n_alpha).cpu()
        
        fig = plt.figure(dpi = self.dpi)
        plt.step(self.alpha_centers.cpu(), relicurve)
        plt.plot((0, 1), (0, 1), 'k:')
        self.tbl.experiment.add_figure("relicurve", fig)
        
    def plot_hist_posts(self):
        hist = self.get_histogram().cpu()
        
        fig = plt.figure(dpi = self.dpi)
        plt.stairs(hist, self.alpha_edges.cpu(), fill=True)
        plt.yscale('log')
        self.tbl.experiment.add_figure("hist_posts", fig)
            
    def plot_hist_sum_posts(self):
        sum_posts = self.get_sum_posts().cpu().numpy()

        fig = plt.figure(dpi = self.dpi)
        hist, _, _ = plt.hist(sum_posts, bins = 50)
#         plt.plot((n_msc, n_msc), (0, hist.max()), 'k:')
        self.tbl.experiment.add_figure("hist_sum_posts", fig)
        
        