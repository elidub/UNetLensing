import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm



def plot_losses_old(mre, title = ''):
    
    tl = mre.state_dict()['training_losses']
    vl = mre.state_dict()['validation_losses']
    epochs = np.arange(1, len(tl)+1)
    
    fig, ax = plt.subplots(1, 1)
            
    ax.plot(epochs, tl, '--', label = f'training loss')
    ax.plot(epochs, vl, '-', label = f'val loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xticks(epochs)
    ax.set_title(title)
    plt.legend()
    plt.show()
    

def plot_losses(mre, title = '', save_name = None):
    
    tl = mre.state_dict()['training_losses']
    vl = mre.state_dict()['validation_losses']
    lr = mre.state_dict()['learning_rates']
    epochs = np.arange(1, len(tl)+1)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    tl_line = ax.plot(epochs, tl, '--', label = 'training loss')
    vl_line = ax.plot(epochs, vl, '-', label = 'valid loss')
    lr_line = ax2.plot(epochs, lr, ':r', label = 'learning rate')

    # Legend
    lns = tl_line + vl_line + lr_line
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    # Labels & title
    ax.set_xlabel('Epoch')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # Integer x-axis ticks
    ax.set_ylabel('Loss')
    ax2.set_ylabel('LR')   
    ax2.set_yscale('log')
    ax.set_title(title)
    
    if save_name is not None:
        plt.savefig(f'../output/figs/loss_{save_name}.png',bbox_inches='tight')
    
    plt.show()
    
def find_nrows(l):
    if l > 1:
        a = np.array([(i, l/i, i+l/i) for i in range(1, l) if (l/i).is_integer()])
        i_min = np.argmin(a[:, -1], axis = 0)
        return int(a[i_min].min()) 
    else:
        return 1

def plt_imshow(plots, nrows = 1, x = 16, y = 8, 
               cmap = [None], cbar = False, ylog = False,
               titles = None, title_size = 12,
               scatter = None, target_coords = None,
               gridlines = None, linspace = None, 
               tl = False, **imkwargs):
    """
    plots: list of what should be platted
    nrows: number of rows
    colobar: if True, a colorbar on each plot is plotted
    size_x, size_y: figsize = (size_x, size_y)
    """
    
    ncols = len(plots) // nrows
    N = len(plots)
    if len(cmap): cmap = cmap*N
    if gridlines is True: gridlines = [1]*N
        
#     colorsstring = 'rbgcmyrbgcmyrbgcmyrbgcmy'

        
    
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize=(x, y))
    
    if ncols == 1:
        axes_flattened = [axes]
    else:
        axes_flattened = axes.flatten()
    
    
    for ax, plot, i in zip(axes_flattened, plots, range(N)):
        if ylog:
            im = ax.imshow(plot, cmap = cmap[i], norm = LogNorm(), **imkwargs)
        else:
            im = ax.imshow(plot, cmap = cmap[i], **imkwargs)
        if titles is not None:
            ax.set_title(titles[i], fontsize = title_size)
        if cbar is True:
            fig.colorbar(im, fraction=0.046, pad=0.04, ax = ax)
        if gridlines is not None:
            if gridlines[i] == 1:
                if 'extent' in imkwargs.keys():
                    ax.set_xticks(np.linspace(imkwargs['extent'][0], imkwargs['extent'][1], (plot.shape)[0]+1), minor = True)
                    ax.set_yticks(np.linspace(imkwargs['extent'][2], imkwargs['extent'][3], (plot.shape)[1]+1), minor = True)
                if linspace is not None:
                    ax.set_xticks(linspace, minor = True)
                    ax.set_yticks(linspace, minor = True)
                    
                ax.grid(which = 'minor', color = 'white',linestyle = '-', linewidth = 2)
        if scatter is not None:
            s = scatter[i]
            for j in range(0, len(s), 2):
                ax.scatter(*s[j:j+2], marker = 'x', c = 'r'
#                            c = [colorsstring[k] for k in range(len(s)+1)]
                          )
    if target_coords is not None:
        for t in target_coords:
            axes_flattened[int(t[0])].scatter(t[2], t[1], marker = 'x', s = 100, color = 'red')
                
      
    if cbar == 'single':
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink = 0.8)
    if tl: # tl stands for Tight Layout 
        plt.tight_layout()
    plt.show()
    
#     return fig, axes
    