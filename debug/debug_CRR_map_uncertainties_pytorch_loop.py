
import sys
import os
import time
from functools import partial
import math
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import scipy.io as sio

import torch
M1 = False

if M1:
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(torch.cuda.current_device()))


# sys.path.append('./../large-scale-UQ/')

import large_scale_UQ as luq
# import optimusprimal as optpr

from convex_reg import utils as utils_cvx_reg
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from large_scale_UQ.utils import to_numpy, to_tensor

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# plt.style.use('dark_background')
# plt.rcParams["font.family"] = "serif"



# Auxiliary functions

# Parameters
# optimization settings
options = {"tol": 1e-5, "iter": 5000, "update_iter": 4999, "record_iters": False}

repo_dir = '/disk/xray0/tl3/repos/large-scale-UQ'
save_dir = repo_dir + '/debug/output/'
savefig_dir = repo_dir + '/debug/figs/'

# Prior parameters
lmbd = 1e5
mu = 20

img_name = 'M31'
# Saving names
save_name = '{:s}_256_CRR_lmbd_{:.1e}_mu_{:.1e}'.format(img_name, mu, lmbd)


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
myType = torch.float32
myComplexType = torch.complex64

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

y = phi.dir_op(torch_img).detach().cpu().squeeze().numpy()

# Generate noise
rng = np.random.default_rng(seed=0)
n = rng.normal(0, sigma, y[y!=0].shape)
# Add noise
y[y!=0] += n

# Observation
torch_y = torch.tensor(np.copy(y), device=device, dtype=myComplexType).reshape((1,) + img.shape)
x_init = torch.abs(phi.adj_op(torch_y))


# Define the likelihood
g = luq.operators.L2Norm_torch(
    sigma=sigma,
    data=torch_y,
    Phi=phi,
)
# g.beta = 1.0 / sigma ** 2

# Define real prox
f = luq.operators.RealProx_torch()


## Load the CRR regulariser
# device = 'cuda:0'
torch.set_grad_enabled(False)
torch.set_num_threads(4)

sigma_training = 5
t_model = 5
dir_name = '/disk/xray0/tl3/repos/convex_ridge_regularizers/trained_models/'
exp_name = f'Sigma_{sigma_training}_t_{t_model}/'
model = utils_cvx_reg.load_model(
    dir_name+exp_name, 'cuda:0', device_type='gpu'
)

print(f'Numbers of parameters before prunning: {model.num_params}')
model.prune()
print(f'Numbers of parameters after prunning: {model.num_params}')

L = model.L.detach().cpu().squeeze().numpy()
print(f"Lipschitz bound {L:.3f}")

# [not required] intialize the eigen vector of dimension (size, size) associated to the largest eigen value
model.initializeEigen(size=100)
# compute bound via a power iteration which couples the activations and the convolutions
model.precise_lipschitz_bound(n_iter=100)
# the bound is stored in the model
L = model.L.data.item()
print(f"Lipschitz bound {L:.3f}")


# y_torch = torch.tensor(y, device=device, dtype=myType, requires_grad=False).reshape((1,1) + y.shape) # .to(torch.float32)
# x_torch = torch.tensor(x.copy(), device=device, dtype=myType, requires_grad=False).reshape((1,1) + x.shape) 


# Gradient descent
# x_init = np.real(phi.adj_op(y))
# x_init_torch = torch.tensor(x_init, device=device, dtype=myType, requires_grad=False).reshape((1,1) + x_init.shape) 



# Compute stepsize
alpha = 1. / ( 1. + g.beta + mu * lmbd * L)

# initialization
x_hat = torch.clone(x_init)
z = torch.clone(x_init)
t = 1


for it_2 in range(options['iter']):
    x_hat_old = torch.clone(x_hat)
    # grad = g.grad(z.squeeze()) +  lmbd * model(mu * z)
    x_hat = z - alpha *(
        g.grad(z) + lmbd * model(mu * z)
    )
    # Positivity constraint
    x_hat =  torch.real(x_hat)
    # possible constraint, AGD becomes FISTA
    # e.g. if positivity
    # x = torch.clamp(x, 0, None)
    
    t_old = t 
    t = 0.5 * (1 + math.sqrt(1 + 4*t**2))
    z = x_hat + (t_old - 1)/t * (x_hat - x_hat_old)

    # relative change of norm for terminating
    res = (torch.norm(x_hat_old - x_hat)/torch.norm(x_hat_old)).item()

    if res < options['tol']:
        print("[GD] converged in %d iterations"%(it_2))
        break

    if it_2 % options['update_iter'] == 0:
        print(
            "[GD] %d out of %d iterations, tol = %f" %(            
                it_2,
                options['iter'],
                res,
            )
        )


# %%
np_x_init = to_numpy(x_init)
np_x = np.copy(x)
np_x_hat = to_numpy(x_hat)

images = [np_x, np_x_init, np_x_hat, np_x - np.abs(np_x_hat)]
labels = ["Truth", "Dirty", "Reconstruction", "Residual (x - x^hat)"]

fig, axs = plt.subplots(1,4, figsize=(20,8), dpi=200)
for i in range(4):
    im = axs[i].imshow(images[i], cmap='cubehelix', vmax=np.nanmax(images[i]), vmin=np.nanmin(images[i]))
    divider = make_axes_locatable(axs[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    if i == 0:
        stats_str = '\nRegCost {:.3f}'.format(model.cost(to_tensor(mu * images[i], device=device))[0].item())
    if i > 0:   
        stats_str = '\n(PSNR: {:.2f}, SNR: {:.2f},\nSSIM: {:.2f}, RegCost: {:.3f})'.format(
            psnr(np_x, images[i], data_range=np_x.max()-np_x.min()),
            luq.utils.eval_snr(x, images[i]),
            ssim(np_x, images[i], data_range=np_x.max()-np_x.min()),
            model.cost(to_tensor(mu * images[i], device=device))[0].item(),
            )
    labels[i] += stats_str
    axs[i].set_title(labels[i], fontsize=16)
    axs[i].axis('off')
plt.savefig('{:s}{:s}_lmbd_{:.1e}_optim_MAP.pdf'.format(savefig_dir, img_name, lmbd))
plt.close()




## Compute HPD region
alpha_prob = 0.05
N = np_x_hat.size
tau_alpha = np.sqrt(16*np.log(3/alpha_prob))

# prior_fun = lambda model, x_hat, mu, lambda_param : (lambda_param/mu) * model.cost(mu*torch.tensor(x_hat, device=device, dtype=myType, requires_grad=False))

def _reg_fun(x_hat, model, mu, lmbd):
    return (lmbd/mu) * to_numpy(model.cost(mu * to_tensor(x_hat, device, myType)))

def likelihood_fun(x_hat):
    return to_numpy(g.fun(to_tensor(x_hat)))

reg_fun = partial(_reg_fun, model=model, mu=mu, lmbd=lmbd)

# gamma_alpha = g.fun(x_hat_np) + (lmbd/mu) * model.cost(mu*torch.tensor(x_hat, device=device, dtype=myType, requires_grad=False)).detach().cpu().squeeze().numpy() + tau_alpha*np.sqrt(N) + N

# Define bound
gamma_alpha = likelihood_fun(np_x_hat) + reg_fun(np_x_hat) + tau_alpha*np.sqrt(N) + N
# Define potential function
loss_fun = lambda x_map : likelihood_fun(x_map) + reg_fun(x_map)


# Compute HPD region

print(
    'f(x_map): ', likelihood_fun(np_x_hat),
    ', g(x_map): ', reg_fun(np_x_hat),
    'tau_alpha*np.sqrt(N): ', tau_alpha*np.sqrt(N),
    'N: ', N,
)
print('N: ', N)
print('tau_alpha: ', tau_alpha)
print('gamma_alpha: ', gamma_alpha)


opt_params = {
    'lmbd': lmbd,
    'mu': mu,
    'sigma_training': sigma_training,
    't_model': t_model,
    'sigma_noise': sigma,
    'opt_tol': options['tol'],
    'opt_max_iter': options['iter'],
}
hpd_results = {
    'alpha': alpha_prob,
    'gamma_alpha': gamma_alpha,
    'f_xmap': likelihood_fun(np_x_hat),
    'g_xmap': reg_fun(np_x_hat),
    'h_alpha_N': tau_alpha*np.sqrt(N) + N,
}
save_dic = {
    'x_map': np_x_hat,
    'opt_params': opt_params,
    'hpd_results': hpd_results,
}
saving_path = save_dir + '{:s}_results.npy'.format(save_name)
# We will overwrite the dict with new results
if os.path.isfile(saving_path):
    os.remove(saving_path)
np.save(saving_path, save_dic, allow_pickle=True)


# Plot computed credible region upper bound
upper_bounds = lambda alpha_prob : likelihood_fun(np_x_hat) + reg_fun(np_x_hat) + np.sqrt(16*np.log(3/alpha_prob)) * np.sqrt(N) + N
plot_alphas = np.linspace(0.01, 0.99, 99)

plt.figure()
plt.plot(plot_alphas, upper_bounds(plot_alphas))
plt.title('f(xmap)={:.1f}\ng(xmap)={:.1f}'.format(
    likelihood_fun(np_x_hat), reg_fun(np_x_hat))
)
plt.savefig('{:s}{:s}_gamma.pdf'.format(savefig_dir, save_name))
plt.close()
# plt.show()

# Compute the LCI
superpix_sizes = [32, 16, 8, 4, 2, 1]
LCI_iters = 200
LCI_tol = 1e-4
LCI_bottom = -10
LCI_top = 10

error_p_arr = []
error_m_arr = []
computing_time = []

for superpix_size in superpix_sizes:

    pr_time_1 = time.process_time()
    wall_time_1 = time.time()

    error_p, error_m, mean = luq.map_uncertainty.create_local_credible_interval(
    x_sol=np_x_hat,
    region_size=superpix_size,
    function=loss_fun,
    bound=gamma_alpha,
    iters=LCI_iters,
    tol=LCI_tol,
    bottom=LCI_bottom,
    top=LCI_top,
    )
    error_length = error_p - error_m

    pr_time_2 = time.process_time()
    wall_time_2 = time.time()

    error_p_arr.append(np.copy(error_p))
    error_m_arr.append(np.copy(error_m))
    computing_time.append((
        pr_time_2 - pr_time_1, 
        wall_time_2 - wall_time_1
    ))

    vmin = np.min((np_x, np_x_init, np_x_hat))
    vmax = np.max((np_x, np_x_init, np_x_hat))
    # err_vmax= 0.6
    cmap='cubehelix'

    plt.figure(figsize=(28,12))
    plt.subplot(241)
    plt.imshow(np_x, cmap=cmap, vmin=vmin, vmax=vmax);plt.colorbar()
    plt.title('Ground truth')
    plt.subplot(242)
    plt.imshow(np_x_init, cmap=cmap, vmin=vmin, vmax=vmax);plt.colorbar()
    plt.title('Dirty')
    plt.subplot(243)
    plt.imshow(np_x_hat, cmap=cmap, vmin=vmin, vmax=vmax);plt.colorbar()
    plt.title('MAP estimator')
    plt.subplot(244)
    plt.imshow(np_x - np_x_hat, cmap=cmap);plt.colorbar()
    plt.title('Ground truth - MAP estimator')
    plt.subplot(245)
    plt.imshow(error_length, cmap=cmap);plt.colorbar()
    plt.title('LCI (max={:.5f})\n (mean={:.5f})'.format(
        np.max(error_length), np.mean(error_length))
    )
    plt.subplot(246)
    plt.imshow(error_length - np.mean(error_length), cmap=cmap);plt.colorbar()
    plt.title('LCI - <LCI>')
    plt.subplot(247)
    plt.imshow(mean, cmap=cmap);plt.colorbar()
    plt.title('Mean')
    plt.savefig(
        '{:s}{:s}_pixSize_{:d}.pdf'.format(savefig_dir, save_name, superpix_size)
    )
    plt.close()
    # plt.show()


LCI_params ={
    'iters': LCI_iters,
    'tol': LCI_tol,
    'bottom': LCI_bottom,
    'top': LCI_top,
}
save_dic = {
    'x_map': np_x_hat,
    'opt_params': opt_params,
    'hpd_results': hpd_results,
    'error_p_arr': error_p_arr,
    'error_m_arr': error_m_arr,
    'computing_time': computing_time,
    'superpix_sizes': superpix_sizes,
    'LCI_params': LCI_params,
}
# We will overwrite the dict with new results
if os.path.isfile(saving_path):
    os.remove(saving_path)

np.save(saving_path, save_dic, allow_pickle=True)

print('Bye')

