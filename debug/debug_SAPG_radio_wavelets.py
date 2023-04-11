# %%
import sys
import os
import time
import math
from functools import partial
import numpy as np
from astropy.io import fits
import scipy.io as sio

import torch
M1 = False

if M1:
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(torch.cuda.current_device()))


import large_scale_UQ as luq

from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from large_scale_UQ.utils import to_numpy, to_tensor

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# %%
options = {"tol": 1e-5, "iter": 5000, "update_iter": 50, "record_iters": False}


# %%
repo_dir = '/Users/tl/Documents/research/repos/proj-convex-UQ/large-scale-UQ'
# repo_dir = '/disk/xray0/tl3/repos/large-scale-UQ'
save_dir = repo_dir + '/notebooks/SAPG/output/'
savefig_dir = repo_dir + '/notebooks/SAPG/figs/'

# optimization settings
wavs =  ["db8"]# ["db1", "db4"]                                     # Wavelet dictionaries to combine
levels = 4 # 3                                               # Wavelet levels to consider [1-6]
reg_param = 50. # 2.e-3
img_name = 'M31'

# Saving names
save_name = '{:s}_256_wavelet_SAPG-{:s}_{:d}_reg_{:.1f}'.format(
    img_name, wavs[0], levels, reg_param
)


# Load img
img_path = repo_dir + '/data/imgs/{:s}.fits'.format(img_name)
img_data = fits.open(img_path, memmap=False)

# Loading the image and cast it to float
img = np.copy(img_data[0].data)[0,:,:].astype(np.float64)
# Flipping data
img = np.flipud(img)

# Aliases
x = img
ground_truth = img
 
 
# Load op from X Cai
op_mask = sio.loadmat(repo_dir + '/data/operators_masks/fourier_mask.mat')['Ma']

# Matlab's reshape works with 'F'-like ordering
mat_mask = np.reshape(np.sum(op_mask, axis=0), (256,256), order='F').astype(bool)

# Define my torch types
myType = torch.float64
myComplexType = torch.complex128

torch_img = torch.tensor(np.copy(img), dtype=myType, device=device).reshape((1,1) + img.shape)
dim = img.shape[0]

# A mock radio imaging forward model with half of the Fourier coefficients masked
# Use X. Cai's Fourier mask
phi = luq.operators.MaskedFourier_torch(
    dim=dim, 
    ratio=0.5 ,
    mask=mat_mask,
    norm='ortho',
    device=device
)

# Define X Cai noise level
sigma = 0.0024
sigma2 = sigma**2

sigma_GT = np.copy(sigma)
sigma2_GT = np.copy(sigma2)

y = phi.dir_op(torch_img).detach().cpu().squeeze().numpy()

# Generate noise
rng = np.random.default_rng(seed=0)
n = rng.normal(0, sigma, y[y!=0].shape)
# Add noise
y[y!=0] += n

# Observation
torch_y = torch.tensor(np.copy(y), device=device, dtype=myComplexType).reshape((1,) + img.shape)
x_init = torch.abs(phi.adj_op(torch_y))



# %%
sigma2

# %%

# Define the likelihood
g = luq.operators.L2Norm_torch(
    sigma=sigma,
    data=torch_y,
    Phi=phi,
)
# g.beta = 1.0 / sigma ** 2

# Define real prox
f = luq.operators.RealProx_torch()


# Define the wavelet dict
# Define the l1 norm with dict psi
# gamma = torch.max(torch.abs(psi.dir_op(y_torch))) * reg_param
psi = luq.operators.DictionaryWv_torch(wavs, levels)

h = luq.operators.L1Norm_torch(1., psi, op_to_coeffs=True)
gamma = h._get_max_abs_coeffs(h.dir_op(torch.clone(x_init))) # * reg_param
h.gamma = gamma
h.beta = 1.0


# %%


# %%

# Define sigma bounds
min_sigma2 = torch.tensor(1e-10, device=device, dtype=myType)
max_sigma2 = torch.tensor(1e1, device=device, dtype=myType)
sigma2_init = torch.tensor(1e-5, device=device, dtype=myType)



# %%
# Negative log-likelihood -logp(t|x,sigma^2)
f = lambda _x, sigma2: g.fun(_x, sigma2=sigma2)

# --- Gradient w.r.t. sigma^2
dimx = x.size
alpha_homogenious = 1
df_wrt_sigma2 = lambda _x, sigma2: torch.real(g.grad_sigma2(_x, sigma2=sigma2)) - dimx / (alpha_homogenious * sigma2)
# Note: The second part corresponds to the normalisation constant Z of the posterior

# --- Gradient w.r.t. x
df_wrt_x = lambda _x, sigma2: torch.real(g.grad(_x, sigma2=sigma2))


# %%
# Define prior
fun_prior = lambda _x : h._fun_coeffs(h.dir_op(torch.clone(_x)))
sub_op = lambda _x1, _x2 : _x1 - _x2

# proximity operator
prox_prior_cai = lambda _x, lmbd : torch.clone(_x) + h.adj_op(h._op_to_two_coeffs(
    h.prox(h.dir_op(torch.clone(_x)), lmbd),
    h.dir_op(torch.clone(_x)), sub_op
))
prox_prior = lambda _x, lmbd : h.adj_op(h.prox(h.dir_op(torch.clone(_x)), lmbd))

# gradient of the prior
gradg = lambda x, lam, lambda_prox: torch.real(x - prox_prior_cai(x,lam)) / lambda_prox    


# %%
# Log of posterior distribution
logPi = lambda _x, sigma2, theta: (- f(_x, sigma2) - theta * fun_prior(_x))


# %%

## Lipschitz Constants

# --- Maximum eigenvalue of operator A. Norm of blurring operator.
g._compute_lip_constant()
AAt_norm =  g.beta * (g.sigma**2) 

# Lipshcitz constant of f.
Lp_fun = lambda sigma2: AAt_norm**2 / sigma2  
# L_f =  min(Lp_fun(min_sigma2), Lp_fun(max_sigma2))
L_f= Lp_fun(sigma2_init)

# --- regularization parameter of proximity operator (\lambda).
lambdaMax = 2
lambda_prox = min((1/L_f), lambdaMax)   
# --- end

# --- Lipshcitz constant of g.
L_g =  1/lambda_prox 

# --- Lipshcitz constant of g + f
L =  L_f + L_g
# --- end

# --- Stepsize of MCMC algorithm.
gamma = 0.98*1/L
# --- end


# %%
print('L: ', L)
print('gamma: ', gamma)
print('lambda_prox: ', lambda_prox)



# %%
# --- Initialization of parameter theta
th_init = 1e3 # 0.01

# --- Admisible set for \theta (min and max values).
min_th = 1e-1 # 0.001
max_th = 1e6 # 1

# --- define stepsize delta 
d_exp = 0.8
delta = lambda _i: (_i**(-d_exp)) / dimx 

# --- constant to tune the stepsize of each parameter
c_theta = 1e2 # 1e1
c_sigma2 = 1e5


# %%
# --- Warmup period for the MCMC sampling
warmupSteps = 1000

# --- total number of iterations for the optimization algorithm on theta
total_iter = 7500

# --- burn-in period for the optimization algorithm on theta
burnIn = int(total_iter * 0.8)

# %%
# --- Initialization of the warm-up chain
# X_wu = y.to(device).detach().clone()
x_init = torch.abs(phi.adj_op(torch_y))
X_wu = torch.clone(x_init)

#Run MYULA sampler with fix theta and fix sigma^2 to warm up the markov chain

fix_sigma2 = torch.tensor(sigma2_GT, device=device, dtype=myType) #sigma2_init
fix_theta = th_init

print('Running Warm up     \n')

for k in tqdm(range(1,warmupSteps)):
    # --- Gradients
    gradf_X_wu = df_wrt_x(X_wu, fix_sigma2)
    gradg_X_wu = gradg(X_wu, lambda_prox*fix_theta, lambda_prox)
    # --- end (gradients)
    
    # --- MYULA warm-up
    X_wu =  X_wu - gamma*gradg_X_wu - gamma*gradf_X_wu + math.sqrt(2*gamma)*torch.randn_like(X_wu)
    # --- end (warm-up)


# %%
np_sigma2_GT = np.copy(sigma2_GT)
sigma2_GT = torch.tensor(sigma2_GT, device=device, dtype=myType)

# %%
# Keeping track of the reg. parameter's trace
theta_trace = torch.zeros(total_iter)
theta_trace[0] = th_init

sigma2_trace= torch.zeros(total_iter)
sigma2_trace[0]= sigma2_init

# We work on a logarithmic scale, so we define an axiliary variable 
#eta such that theta=exp{eta}. 

eta_init = math.log(th_init)
min_eta = math.log(min_th)
max_eta = math.log(max_th)

eta_trace = torch.zeros(total_iter)
eta_trace[0] = eta_init

# Stop criteria (relative change tolerance) for the proximal gradient algorithm

stopTol=1e-5


# %%
print('\nRunning SAPG algorithm     \n')
# We want to keep track of two traces: the log-likelihood and the TV function to monitor the behaviour of the algorithm.

logPiTraceX = []      # to monitor convergence
g_trace = []          # to monitor how the regularisation function evolves

mean_theta =[]
mean_sigma2 = []
Grad_sigma2 = []

X = X_wu.clone()       # start MYULA markov chain from last sample after warmup

update_iter = 100

# for k in tqdm(range(1,total_iter)): 
for k in range(1,total_iter): 

    ################################################################################
    # MCMC SAMPLER
    ################################################################################

    # Number of samples

    m = 1

    # If we run the MCMC sampler for m times to get m samples X_m, therefore we need to average 
    # gradients w.r.t. \theta and \sigma^2 before the update
    g_mcmc_trace = torch.zeros(m, device=device) # .to(device) 
    grad_sigma2_trace = torch.zeros(m, device=device) # .to(device) 
    
    #Sample from posterior with MYULA:
    
    for ii in range(m):

        # Calculate the gradient related to g for the current theta
        gradgX = gradg(X,lambda_prox*theta_trace[k-1],lambda_prox)  
        
        # --- Calculate the gradient related to f for the current theta
        # gradfX = df_wrt_x(X,sigma2_trace[k-1])
        gradfX = df_wrt_x(X,sigma2_GT)
        # --- end
        
        # --- MYULA update
        X =  X - gamma*gradgX - gamma*gradfX + math.sqrt(2*gamma)*torch.randn_like(X)
        #X = torch.clamp(X,0,255)
        # --- end
        
        # --- Gardients w.r.t parameters
        g_mcmc_trace[ii] = fun_prior(X)
        # grad_sigma2_trace[ii] = df_wrt_sigma2(X, sigma2_trace[k-1])
        # --- end
        
    # --- Save current state to monitor convergence
    # logPiTraceX.append(logPi(X, sigma2_trace[k-1],theta_trace[k-1]).cpu().numpy())
    logPiTraceX.append(logPi(X, sigma2_GT, theta_trace[k-1]).cpu().numpy())
    g_trace.append(g_mcmc_trace[-1].cpu().numpy())
    # --- end (monitoring)
    
    # ################################################################################
    #  PROJECTED GRADIENT ALGORITHM
    # ################################################################################

    # Update eta and theta. It should be underlined that we work on the logarithmic
    # scale for numerical stability

    # --- update \eta and \theta
    etak = eta_trace[k-1] + c_theta * delta(k)  * (dimx / theta_trace[k-1] - torch.mean(g_mcmc_trace)) * torch.exp(eta_trace[k-1]) 

    # project \eta onto the admissible set of value
    eta_trace[k] = min(max(etak,min_eta),max_eta)
    
    # Save the value of theta
    theta_trace[k] = torch.exp(eta_trace[k])
    # --- end (update)
    
    # --- Update sigma^2
    # sigma2_k = sigma2_trace[k-1] + c_sigma2 * delta(k) * torch.mean(grad_sigma2_trace)

    # sigma2_trace[k] = min(max(sigma2_k, min_sigma2), max_sigma2)
    sigma2_trace[k] = sigma2_GT
    # --- end (update)
    
    # -- 
    # Grad_sigma2.append(df_wrt_sigma2(X, sigma2_trace[k-1]).cpu().numpy())
    if k % update_iter == 0:
        # print(f"iter = {k} \t Theta = {theta_trace[k]} \t sigma2 = {sigma2_trace[k]}\n")
        print(f"iter = {k} \t Theta = {theta_trace[k]}\n")
   
    # --- Check stop criteria. If relative error is smaller than op.stopTol stop
    
    if k>burnIn+1:
        mean_theta.append(torch.mean(theta_trace[burnIn:(k+1)]).cpu().numpy())
        # mean_sigma2.append(torch.mean(sigma2_trace[burnIn:(k+1)]).cpu().numpy())
        
        relErrTh1 = torch.abs(torch.mean(theta_trace[burnIn:(k+1)]) - torch.mean(theta_trace[burnIn:k])) / torch.mean(theta_trace[burnIn:k])
        
        # relErrSi1 = torch.abs(torch.mean(sigma2_trace[burnIn:(k+1)]) - torch.mean(sigma2_trace[burnIn:k])) / torch.mean(sigma2_trace[burnIn:k])

        # if (relErrTh1<stopTol) and (relErrSi1<stopTol) and 1 == 2 :
        if (relErrTh1<stopTol) and 1 == 2 :    
            print("Toleration reached!")
            break
     # --- end (stop criteria)       

# --- Collecting data
last_samp = k

logPiTraceX = logPiTraceX[:last_samp+1]
gXTrace = g_trace[:last_samp+1]

theta_EB = torch.exp(torch.mean(eta_trace[burnIn:last_samp+1]))
last_theta = theta_trace[last_samp]
thetas = theta_trace[:last_samp+1]

# sigma2_EB = torch.mean(sigma2_trace[burnIn:last_samp+1])
# sigmas=sigma2_trace[:last_samp+1]



# %%
# Estimated theta:  tensor(5.2098)
# Last theta:  tensor(5.0593)


# %%
print("Estimated theta: ", theta_EB)
print("Last theta: ", last_theta)
# print("Estimated value of sigma2 ",sigma2_EB, sigma**2)

# %%
# Plot the results

plot1 = plt.figure()
plt.plot(thetas[:].cpu().numpy(),linestyle="-")
plt.xlabel("$Iterations$")
plt.ylabel("$θ$")

plot1 = plt.figure()
plt.plot( logPiTraceX,linestyle="-")
plt.xlabel("$Iterations$")
plt.ylabel("$log(p(x|y))$")

plot1 = plt.subplots()
plt.plot( gXTrace[:],linestyle="-")
plt.plot( 256*256/thetas[:].cpu().numpy(),linestyle="-")
plt.xlabel("$Iterations$")
plt.ylabel("$g(x) versus d/theta$")

# fig, ax = plt.subplots()
# ax.plot(sigmas.cpu().numpy(),linestyle="-",label="$σ_{n}^{2}$")
# plt.axhline(y=sigma.cpu().numpy()**2, color='r', linestyle='--',label="$σ_{\dagger}^{2}$")
# #plt.ylim(min_sigma2, max_sigma2)
# ax.set_xlabel("$Iterations\,\,(n)$")
# ax.set_ylabel("$sigma^{2}$")
# ax.legend()


# %%
plot1 = plt.figure()
plt.plot(thetas[1000:].cpu().numpy(),linestyle="-")
plt.xlabel("$Iterations$")
plt.ylabel("$θ$")

# %%


# %%



