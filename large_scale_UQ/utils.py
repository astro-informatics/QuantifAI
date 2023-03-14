
# Utils and helpers

import numpy as np
import torch
import arviz
import matplotlib.pyplot as plt
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
    return round(20*np.log10(num/den), 2)


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

def max_eigenval(A, At, im_size, tol, max_iter, verbose, device):

    with torch.no_grad():

    #computes the maximum eigen value of the compund operator AtA
        
        x = torch.normal(mean=0, std=1,size=(im_size,im_size))[None][None].to(device)
        x = x/torch.norm(torch.ravel(x),2)
        init_val = 1
        
        for k in range(0,max_iter):
            y = A(x)
            x = At(y)
            val = torch.norm(torch.ravel(x),2)
            rel_var = torch.abs(val-init_val)/init_val
            if (verbose > 1):
                print('Iter = {}, norm = {}',k,val)
            
            if (rel_var < tol):
                break
            
            init_val = val
            x = x/val
        
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