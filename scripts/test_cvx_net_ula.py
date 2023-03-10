## Assembled by Teresa Klatzer (TK)
## t.klatzer@sms.ed.ac.uk
##
## with code snippets by TK, Savvas Melidonis, Charlesquin Kemajou, David Thong
## and from https://github.com/axgoujon/convex_ridge_regularizers
## University of Edinburgh // Heriot-Watt University // Project BLOOM

#### small helpers to create the blur operators

import numpy as np
import hdf5storage
import sys
sys.path.append("~/code")
import os
os.chdir("C:/Users/teresa-klatzer/code")

def blur_operators(kernel_len, size, type_blur):

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

    H_FFT = torch.fft.fft2(torch.roll(h, shifts = (-c[0],-c[1]), dims=(0,1)))
    HC_FFT = torch.conj(H_FFT)

    # A forward operator
    A = lambda x: torch.fft.ifft2(torch.multiply(H_FFT,torch.fft.fft2(x[0,0]))).real.reshape(x.shape)
    # A backward operator
    AT = lambda x: torch.fft.ifft2(torch.multiply(HC_FFT,torch.fft.fft2(x[0,0]))).real.reshape(x.shape)

    AAT_norm = max_eigenval(A, AT, nx, 1e-4, int(1e4), 0)

    return A, AT, AAT_norm

def max_eigenval(A, At, im_size, tol, max_iter, verbose):

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

# Welford's algorithm for calculating mean and variance
# Collection of summary statistics of the Markov chain iterates
# 
# call update() when generating a new sample
# call get_mean() or get_var() to obtain the current mean or variance
#
# https://doi.org/10.2307/1266577
class welford: 
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


import matplotlib.pyplot as plt
import numpy as np
import torch
from statsmodels.graphics.tsaplots import plot_acf
import arviz

# ---  Create the necessary function for the autocorrelation plot
def autocor_plots(X_chain, var, method_str, nLags=100):
    '''
    Inputs:
        - X_chain (Matrix): Markov chain
        - var (matrix): Variance of the Markov chain computed in the Fourier domain.
        - M=method_str (string): Name of the Markov chain used
        - nLags (int): Number of lags used for the autocorrelation function.
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
    
    
def ESS(arr):
    '''
    Input: 
        - arr (vector): This vector contains the trace of the Markov chain
    Output:
        - ess (float): effective sample size of the trace.
    '''
    ess = arviz.ess(arr)
    return ess


#### end helpers


### start the script, setup the imaging problem (deblurring)
### use the cvx ridge regularizer

import torch
sys.path.append("convex_ridge_regularizers")

### set seed
torch.manual_seed(0)

# from models import utils
from convex_reg import utils
device = 'cuda:0'
torch.set_grad_enabled(False)
torch.set_num_threads(4)

sigma_training = 25
t = 5
exp_name = f'Sigma_{sigma_training}_t_{t}'
model = utils.load_model(exp_name, device)

print(f'Numbers of parameters before prunning: {model.num_params}')
model.prune()
print(f'Numbers of parameters after prunning: {model.num_params}')

# [not required] intialize the eigen vector of dimension (size, size) associated to the largest eigen value
model.initializeEigen(size=100)
# compute bound via a power iteration which couples the activations and the convolutions
model.precise_lipschitz_bound(n_iter=100)
# the bound is stored in the model
L = model.L.data.item()
print(f"Lipschitz bound {L:.3f}")


im = torch.empty((1, 1, 100, 100), device=device).uniform_()
grad = model.grad(im)# alias for model.forward(im) and hence model(im)

im = torch.empty((1, 1, 100, 100), device=device).uniform_()
cost = model.cost(100*im)


import matplotlib.pyplot as plt
import cv2
import math
img = cv2.resize(cv2.imread("./convex_ridge_regularizers/image/sample.JPG", cv2.IMREAD_GRAYSCALE), (504, 378))
img = img[:377, :377]
img_torch = torch.tensor(img, device=device).reshape((1,1) + img.shape)/255


# try regularizer on a deblurring problem (using 5x5 uniform blur)
# first, setup the operators
A, AT, AAT_norm = blur_operators([5,5], [img.shape[0],img.shape[1]],
                                 "uniform")


# set the desired noise level and add noise to the burred image
sigma = 1/255.
sigma2 = sigma**2
img_torch_blurry = A(img_torch) + sigma * torch.randn_like(img_torch)

# Hyperparameters for the nn regularizer

# found by grid search below
#mu = 5.2
#lmbd = 1600
# computed with fixed mu, for mountain image
mu = 20
lmbd = 1405


# optimization settings
tol = 1e-4
n_iter_max = 300

# stepsize rule
L = model.L 
alpha = 1/( 1 + mu * lmbd * L + AAT_norm/sigma2)

# initialization
# small x - optimization variable, large X sampling variable (see later)
x = torch.clone(img_torch_blurry)
z = torch.clone(img_torch_blurry)
t = 1

for i in range(n_iter_max):
    x_old = torch.clone(x)
    # added deblurring data term here
    x = z - alpha*(AT(A(z) - img_torch_blurry)/sigma2 + lmbd * model(mu * z))
    # possible constraint, AGD becomes FISTA
    # e.g. if positivity
    # x = torch.clamp(x, 0, None)
    
    t_old = t 
    t = 0.5 * (1 + math.sqrt(1 + 4*t**2))
    z = x + (t_old - 1)/t * (x - x_old)

    # relative change of norm for terminating
    res = (torch.norm(x_old - x)/torch.norm(x_old)).item()
    if res < tol:
        break



### Now, try Sampling

#function handles to used for ULA
f = lambda x: lmbd/mu * model.cost(mu * x) + 0.5*torch.norm((A(x) - img_torch_blurry))**2/sigma2
grad_f = lambda x: AT(A(x) - img_torch_blurry)/sigma2 + lmbd * model(mu * x)

#ULA kernel
def ULA_kernel(x, delta):
    return x - delta * grad_f(x) + math.sqrt(2*delta) * torch.randn_like(x)


# Set up sampler
Lip_total = mu * lmbd * L + AAT_norm/sigma2

#step size for ULA
gamma = 1/Lip_total

maxit = 10000
burnin = np.int64(maxit*0.05)
n_samples = np.int64(100)
X = img_torch_blurry.clone()
MC_X = np.zeros((n_samples, X.shape[2], X.shape[3]))
thinned_trace_counter = 0
thinning_step = np.int64(maxit/n_samples)

psnr_values = []
ssim_values = []
logpi_eval = []

# Sample
import time as time
from tqdm.auto import tqdm
from torchmetrics.functional import structural_similarity_index_measure 
from torchmetrics.functional import peak_signal_noise_ratio 

start_time = time.time()
for i_x in tqdm(range(maxit)):

    # Update X
    X = ULA_kernel(X, gamma)

    if i_x == burnin:
        # Initialise recording of sample summary statistics after burnin period
        post_meanvar = welford(X)
        absfouriercoeff = welford(torch.fft.fft2(X).abs())
    elif i_x > burnin:
        # update the sample summary statistics
        post_meanvar.update(X)
        absfouriercoeff.update(torch.fft.fft2(X).abs())

        # collect quality measurements
        current_mean = post_meanvar.get_mean()
        psnr_values.append(peak_signal_noise_ratio(current_mean, img_torch).item())
        ssim_values.append(structural_similarity_index_measure(current_mean, img_torch).item())
        logpi_eval.append(f(X).item())

        # collect thinned trace
        if np.mod(i_x,thinning_step) == 0:
            MC_X[thinned_trace_counter] = X.detach().cpu().numpy()
            thinned_trace_counter += 1

end_time = time.time()
elapsed = end_time - start_time    

current_mean = post_meanvar.get_mean()
current_var = post_meanvar.get_var().detach().cpu().squeeze()


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
ax[0].set_title(f"Clean Image (Reg Cost {model.cost(mu*img_torch)[0].item():.1f})")
ax[0].imshow(img_torch.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[0].set_yticks([])
ax[0].set_xticks([])

ax[1].set_title(f"Blurry Image (Reg Cost {model.cost(mu*img_torch_blurry)[0].item():.1f})")
ax[1].imshow(img_torch_blurry.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[1].set_yticks([])
ax[1].set_xticks([])

ax[2].set_title(f"Deblurred Image MAP (Regularization Cost {model.cost(mu*x)[0].item():.1f}, PSNR: {peak_signal_noise_ratio(x, img_torch).item():.2f})")
ax[2].imshow(x.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax[2].set_yticks([])
ax[2].set_xticks([])

fig, ax = plt.subplots()
ax.set_title("log pi")
ax.plot(np.arange(1,len(logpi_eval)+1), logpi_eval)

fig, ax = plt.subplots()
ax.set_title(f"Deblurred Image MMSE (Regularization Cost {model.cost(mu*current_mean)[0].item():.1f}, PSNR: {peak_signal_noise_ratio(current_mean, img_torch).item():.2f})")
ax.imshow(current_mean.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax.set_yticks([])
ax.set_xticks([])

fig, ax = plt.subplots()
ax.set_title(f"Deblurred Image VAR")
ax.imshow(current_var, cmap="gray")
ax.set_yticks([])
ax.set_xticks([])


autocor_plots(MC_X, current_var, "ULA", nLags=20)

fig, ax = plt.subplots()
ax.set_title("SSIM")
ax.semilogx(np.arange(1,len(ssim_values)+1), ssim_values)

fig, ax = plt.subplots()
ax.set_title("PSNR")
ax.semilogx(np.arange(1,len(psnr_values)+1), psnr_values)



plt.show()

## Now, try hyperparameter tuning



def score(lmbd, mu=20):

    # optimization settings
    tol = 1e-4
    n_iter_max = 100

    # stepsize rule
    L = model.L 
    alpha = 1/( 1 + mu * lmbd * L + AAT_norm/sigma2)

    # initialization
    x = torch.clone(img_torch_blurry)
    z = torch.clone(img_torch_blurry)
    t = 1
    used_iter = 0

    for i in range(n_iter_max):
        x_old = torch.clone(x)
        x = z - alpha*(AT(A(z) - img_torch_blurry)/sigma2 + lmbd * model(mu * z))
        # possible constraint, AGD becomes FISTA
        # e.g. if positivity
        # x = torch.clamp(x, 0, None)
        
        t_old = t 
        t = 0.5 * (1 + math.sqrt(1 + 4*t**2))
        z = x + (t_old - 1)/t * (x - x_old)

        # relative change of norm for terminating
        res = (torch.norm(x_old - x)/torch.norm(x_old)).item()
        if res < tol:
            used_iter = i
            break

    # validator optimizes based on first quantity
    return structural_similarity_index_measure(x, img_torch).item(), peak_signal_noise_ratio(x, img_torch, data_range = 1.0).item(), used_iter


# initialize the validation process
from hyperparameter_tuning.validateCoarseToFine import ValidateCoarseToFine
validator = ValidateCoarseToFine(score, dir_name="./", exp_name="Deblur_Test", gamma_stop=1.05, p1_init=100, p2_init=20, freeze_p2=True)
# run the validation
#validator.run()


