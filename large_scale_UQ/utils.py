
# Utils and helpers

import numpy as np
import torch
import arviz
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.graphics.tsaplots import plot_acf


def to_numpy(z): 
    return z.detach().cpu().squeeze().numpy()

def to_tensor(z, device='cuda', dtype=torch.float):
    return torch.tensor(
        z, device=device, dtype=dtype, requires_grad=False
    ).reshape((1,1) + z.shape)

def eval_snr(x, x_est):
    if np.array_equal(x, x_est):
        return 0
    num = np.sqrt(np.sum(np.abs(x) ** 2))
    den = np.sqrt(np.sum(np.abs(x - x_est) ** 2))
    return round(20 * np.log10(num / den), 2)


def blur_operators(kernel_len, size, type_blur, device):

    nx = size[0]
    ny = size[1]
    if type_blur=='uniform':
        h = torch.zeros(nx,ny).to(device)
        lx = kernel_len[0]
        ly = kernel_len[1]
        h[0:lx,0:ly] = 1/(lx*ly)
        c =  np.ceil((np.array([ly,lx])-1)/2).astype("int64")
    else:
        print("Write more code!")
        raise NotImplementedError

    H_FFT = torch.fft.fft2(torch.roll(h, shifts = (-c[0],-c[1]), dims=(0,1)))
    HC_FFT = torch.conj(H_FFT)

    # A forward operator
    A = lambda x: torch.fft.ifft2(torch.multiply(H_FFT,torch.fft.fft2(x[0,0]))).real.reshape(x.shape)
    # A backward operator
    AT = lambda x: torch.fft.ifft2(torch.multiply(HC_FFT,torch.fft.fft2(x[0,0]))).real.reshape(x.shape)

    AAT_norm = max_eigenval(A, AT, nx, 1e-4, int(1e4), 0, device=device)

    return A, AT, AAT_norm

def max_eigenval(
    A,
    At,
    im_shape,
    tol=1e-4,
    max_iter=np.int64(1e4),
    verbose=0,
    device=torch.device('cpu')
):
    """Computes the maximum eigen value of the compund operator AtA.

    Follows X. Cai article's areas.

    Args:
        A (Callable): Radio image name
        At (Callable): if the area contains phyisical information
        im_shape (list, tuple or np.ndarray): Image shape where `len(im_shape)=2`
        tol (float): Algorithm's tolerance
        max_iter (int): Max number of iterations
        verbose (float): If verbose>0, verbose mode is activated
        device (torch.device): Torch's device

    Returns:
        val (float): max eigenvalue of the AtA operator
    """
    with torch.no_grad():
        x = torch.normal(
            mean=0, std=1, size=im_shape
        )[None][None].to(device)
        x = x / torch.norm(torch.ravel(x), 2)
        init_val = 1
        
        for k in range(max_iter):
            y = A(x)
            x = At(y)
            val = torch.norm(torch.ravel(x), 2)
            rel_var = torch.abs(val - init_val) / init_val
            if (verbose > 1):
                print('Iter = {}, norm = {}', k, val)
            
            if (rel_var < tol):
                break
            
            init_val = val
            x = x / val
        
        if (verbose > 0):
            print('Norm = {}', val)
        
        return val


class welford: 
    """
    Welford's algorithm for calculating mean and variance
    Collection of summary statistics of the Markov chain iterates

    call update() when generating a new sample
    call get_mean() or get_var() to obtain the current mean or variance

    https://doi.org/10.2307/1266577
    """
    def __init__(self, x):
        self.k = 1
        self.M = x.clone()
        self.S = 0

    def update(self, x):
        self.k += 1
        Mnext = self.M + (x - self.M) / self.k
        self.S += (x - self.M)*(x - Mnext)
        self.M = Mnext
    
    def get_mean(self):
        return self.M
    
    def get_var(self):
        #when the number of samples is 1, we divide by self.k 
        if self.k < 2:
            return self.S/(self.k)
        else:
            return self.S/(self.k-1)


def ESS(arr):
    '''
    Input: 
        - arr (vector): This vector contains the trace of the Markov chain
    Output:
        - ess (float): effective sample size of the trace.
    '''
    ess = arviz.ess(arr)
    return ess

# ---  Create the necessary function for the autocorrelation plot
def autocor_plots(X_chain, var, method_str, nLags=100, save_path=None):
    '''
    Inputs:
        - X_chain (Matrix): Markov chain
        - var (matrix): Variance of the Markov chain computed in the Fourier domain.
        - M=method_str (string): Name of the Markov chain used
        - nLags (int): Number of lags used for the autocorrelation function.
        - save_path (str): Path to save the figure. If None, the figure is not saved.
    Outputs:
        - Autocorrelation plots. we also highligthed the effective sample size for each trace.
    '''
    
    # --- Checking whether the samples size is greater than the number of lags considered.
    if X_chain.shape[0] < nLags:
        raise ValueError(f"nLags must be smaller than the number of samples!")
    
   # --- 
    if torch.is_tensor(X_chain):
        X_chain = X_chain.detach().cpu().numpy()
    
    # --- Vectorise the the variance in the Fourier domain
    var_fft = var.reshape(-1,1)
    
    # --- Vectorise the the Markov chain
    X_chain_vec = X_chain.reshape(len(X_chain),-1)
    
    # --- Variance of the Markov not in the spatial
    var_sp = np.var(X_chain_vec, axis = 0)
    
    # --- lower trace of the Markov chain
    trace_elem_max_variance = X_chain_vec[:,np.argmax(var_sp)]
    
     # --- Faster trace of the Markov chain
    trace_elem_min_variance = X_chain_vec[:,np.argmin(var_sp)]
    
     # --- Medium-speed trace of the Markov chain
    ind_medium = np.argsort(var_sp)[len(var_sp)//2]
    trace_elem_median_variance = X_chain_vec[:,ind_medium]
    
    # --- effective sample size
    e_slow = ESS(trace_elem_max_variance.reshape(-1))
    e_fast = ESS(trace_elem_min_variance.reshape(-1))
    e_med  = ESS(trace_elem_median_variance.reshape(-1))

    # --- Here we are generating the autocorrelation function for these three traces: lower, medium and faster.
    fig,ax = plt.subplots(figsize=(15,10))
    plot_acf(trace_elem_median_variance,ax=ax,label=f'Median-speed, ESS = {e_med: .2f}',alpha=None,lags=nLags)
    plot_acf(trace_elem_max_variance,ax=ax,label=f'Slowest, ESS = {e_slow: .2f}',alpha=None,lags=nLags)
    plot_acf(trace_elem_min_variance,ax=ax,label=f'Fastest, ESS = {e_fast: .2f}',alpha=None,lags=nLags)
    handles, labels= ax.get_legend_handles_labels()
    handles=handles[1::2]
    labels =labels[1::2]
    ax.set_title(method_str)
    ax.set_ylabel("ACF")
    ax.set_xlabel("lags")
    ax.set_ylim([-1.1,1.3])
    ax.legend(handles=handles, labels=labels,loc='best',shadow=True, numpoints=1)

    if save_path is not None:
        plt.savefig(save_path)


def plot_summaries(
        x_ground_truth,
        x_dirty,
        post_meanvar,
        post_meanvar_absfourier,
        cmap='gray',
        save_path=None
):
    """Plot summaries of the sampling results.
    
    Args:
        x_ground_truth (np.ndarray): Ground truth image.
        x_dirty (np.ndarray): Dirty image.
        post_meanvar (welford instance): Instance of welford with torch variables
            saving the MC samples.
        post_meanvar_absfourier (welford instance): Instance of welford with 
        torch variables saving the absolute values of Fourier coefficients of MC samples.
        save_path (str): Path to save the figure. If None, the figure is not saved.
    """
    
    post_mean_numpy = post_meanvar.get_mean().detach().cpu().squeeze().numpy()
    post_var_numpy = post_meanvar.get_var().detach().cpu().squeeze().numpy()

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize = (18,10))
    fig.tight_layout(pad=.01)
    
    # --- Ground truth
    im = axes[0,0].imshow(x_ground_truth, cmap=cmap)
    axes[0,0].set_title('Ground truth image')
    axes[0,0].axis('off')
    divider = make_axes_locatable(axes[0,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # --- Blurred
    im = axes[0,1].imshow(x_dirty, cmap=cmap)
    axes[0,1].set_title('Dirty image')
    axes[0,1].axis('off')
    divider = make_axes_locatable(axes[0,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # --- MMSE
    im = axes[0,2].imshow(post_mean_numpy, cmap=cmap)
    axes[0,2].set_title('x - posterior mean')
    axes[0,2].axis('off')
    divider = make_axes_locatable(axes[0,2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # --- Variance
    im = axes[0,3].imshow(post_var_numpy, cmap=cmap)
    axes[0,3].set_title('x - posterior variance')
    axes[0,3].axis('off')
    divider = make_axes_locatable(axes[0,3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # --- MMSE / Var
    im = axes[1,0].imshow(post_mean_numpy/np.sqrt(post_meanvar.get_var().detach().cpu().squeeze().numpy()), cmap=cmap)
    axes[1,0].set_title('x - posterior mean/posterior SD')
    axes[1,0].axis('off')
    divider = make_axes_locatable(axes[1,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # --- Var / MMSE
    im = axes[1,1].imshow(np.sqrt(post_var_numpy)/post_mean_numpy,cmap=cmap)
    axes[1,1].set_title('x - Coefs of variation')
    axes[1,1].axis('off')
    divider = make_axes_locatable(axes[1,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # --- Mean Fourier coefs
    im = axes[1,2].imshow(torch.log(post_meanvar_absfourier.get_mean()).detach().cpu().squeeze().numpy())
    axes[1,2].set_title('Mean coefs (log-scale)')
    axes[1,2].axis('off')
    divider = make_axes_locatable(axes[1,2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    # --- Variance Fourier coefs
    im = axes[1,3].imshow(torch.log(post_meanvar_absfourier.get_var()).detach().cpu().squeeze().numpy())
    axes[1,3].set_title('Var coefs (log-scale)')
    axes[1,3].axis('off')
    divider = make_axes_locatable(axes[1,3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    if save_path is not None:
        plt.savefig(save_path)
