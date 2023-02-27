
import sys
import os
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import scipy.io as sio

sys.path.append('./../large-scale-UQ/')

import large_scale_UQ as luq
import optimusprimal as optpr

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"



# Auxiliary functions

def eval_snr(x, x_est):
    if np.array_equal(x, x_est):
        return 0
    num = np.sqrt(np.sum(np.abs(x) ** 2))
    den = np.sqrt(np.sum(np.abs(x - x_est) ** 2))
    return round(20*np.log10(num/den), 2)

# Parameters
options = {"tol": 1e-5, "iter": 3500, "update_iter": 50, "record_iters": False}
save_dir = './debug/output/'
savefig_dir = './debug/figs/'
# optimization settings
lmbd = 100
mu = 20
img_name = 'M31'
save_name = '{:s}_256_CRR_mu{:d}_lmbd{:d}'.format(img_name, mu, lmbd)


# Load img
img_path = '/Users/tliaudat/Documents/postdoc/github/large-scale-UQ/data/imgs/{:s}.fits'.format(img_name)
img_data = fits.open(img_path, memmap=False)

# Loading the image and cast it to float
img = np.copy(img_data[0].data)[0,:,:].astype(np.float64)
# Flipping data
img = np.flipud(img)

# Aliases
x = img
ground_truth = img
 
# Load op from X Cai
op_mask = sio.loadmat('/Users/tliaudat/Documents/postdoc/github/large-scale-UQ/data/operators_masks/fourier_mask.mat')['Ma']

# Matlab's reshape works with 'F'-like ordering
mat_mask = np.reshape(np.sum(op_mask, axis=0), (256,256), order='F').astype(bool)




dim = img.shape[0]

# A mock radio imaging forward model with half of the Fourier coefficients masked
phi = luq.operators.MaskedFourier(dim, 0.5 , norm='ortho')
# Use X. Cai's Fourier mask
phi.mask = mat_mask

# Simulate mock noisy observations y
y = phi.dir_op(img)
ISNR = 25
sigma = np.sqrt(np.mean(np.abs(y)**2)) * 10**(-ISNR/20)

# Using the same noise as in X Cai
sigma = 0.0024

# Add noise
rng = np.random.default_rng(seed=0)
n = rng.normal(0, sigma, y.shape)

# Simulate mock noisy observations
y += n
# Define init img
x_init = np.abs(phi.adj_op(y))


## Load the CRR regulariser
import torch
from convex_reg import utils
from tqdm import tqdm

gpu = False

if gpu:
    device = 'cuda:0'
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
else:
    device = 'cpu'
    
sigma_training = 5
t_model = 5
dir_name = '/Users/tliaudat/Documents/postdoc/github/convex_ridge_regularizers/trained_models/'
exp_name = f'Sigma_{sigma_training}_t_{t_model}/'
model = utils.load_model(dir_name+exp_name, device, gpu=gpu)

print(f'Numbers of parameters before prunning: {model.num_params}')
model.prune()
print(f'Numbers of parameters after prunning: {model.num_params}')



myType = torch.float32

# y_torch = torch.tensor(y, device=device, dtype=myType, requires_grad=False).reshape((1,1) + y.shape) # .to(torch.float32)
# x_torch = torch.tensor(x.copy(), device=device, dtype=myType, requires_grad=False).reshape((1,1) + x.shape) 

# Define the grad
g = optpr.grad_operators.l2_norm(sigma, y, phi)
g.beta = 1.0 / sigma ** 2

# Real prox 
f = optpr.prox_operators.real_prox()
# f = None


# Gradient descent
x_init = np.real(phi.adj_op(y))
# x_init_torch = torch.tensor(x_init, device=device, dtype=myType, requires_grad=False).reshape((1,1) + x_init.shape) 

# Helper functions
to_tensor = lambda _z : torch.tensor(_z, device=device, dtype=myType, requires_grad=False).reshape((1,1) + _z.shape)
to_numpy = lambda _z : _z.detach().squeeze().numpy()

# stepsize rule
L = to_numpy(model.L)
alpha = 1. / ( 1. + g.beta + mu * lmbd * L)

# initialization
x_hat = np.copy(x_init)
z = np.copy(x_init)
t = 1



for it in range(options['iter']):
    x_hat_old = np.copy(x_hat)
    # grad = g.grad(z.squeeze()) +  lmbd * model(mu * z)
    x_hat = z - alpha *(
        g.grad(z) + lmbd * to_numpy(model(to_tensor(mu * z)))
    )
    x_hat =  np.real(x_hat)
    # possible constraint, AGD becomes FISTA
    # e.g. if positivity
    # x = torch.clamp(x, 0, None)
    
    t_old = t 
    t = 0.5 * (1 + np.sqrt(1 + 4*t**2))
    z = x_hat + (t_old - 1)/t * (x_hat - x_hat_old)

    # relative change of norm for terminating
    res = (np.linalg.norm(x_hat_old - x_hat)/np.linalg.norm(x_hat_old)) # .item()
    if res < options['tol']:
        print("[GD] converged in %d iterations"%(it))
        break

    if it % options['update_iter'] == 0:
        print(
            "[GD] %d out of %d iterations, tol = %f" %(            
                it,
                options['iter'],
                res,
            )
        )



x_hat_np = x_hat

images = [x, x_init, x_hat_np, x-np.abs(x_hat_np)]
labels = ["Truth", "Dirty", "Reconstruction", "Residual (x - x^hat)"]

fig, axs = plt.subplots(1,4, figsize=(20,8), dpi=200)
for i in range(4):
    im = axs[i].imshow(images[i], cmap='cubehelix', vmax=np.nanmax(images[i]), vmin=np.nanmin(images[i]))
    divider = make_axes_locatable(axs[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    if i == 0:
        stats_str = '\nRegCost {:.3f}'.format(model.cost(to_tensor(mu * images[i]))[0].item())
    if i > 0:   
        stats_str = '\n(PSNR: {:.2f}, SNR: {:.2f},\nSSIM: {:.2f}, RegCost: {:.3f})'.format(
            psnr(ground_truth, images[i], data_range=ground_truth.max()-ground_truth.min()),
            eval_snr(x, images[i]),
            ssim(ground_truth, images[i]),
            model.cost(to_tensor(mu * images[i]))[0].item(),
            )
    labels[i] += stats_str
    axs[i].set_title(labels[i], fontsize=16)
    axs[i].axis('off')
plt.savefig('{:s}{:s}_MAP.pdf'.format(savefig_dir, save_name))
plt.close()
# plt.show()


## Compute HPD region
alpha = 0.01
N = x_hat_np.size
tau_alpha = np.sqrt(16*np.log(3/alpha))


def _reg_fun(x_hat, model, mu, lmbd):
    return (lmbd/mu) * to_numpy(model.cost(to_tensor(mu * x_hat)))

reg_fun = partial(_reg_fun, model=model, mu=mu, lmbd=lmbd)

# Define bound
gamma_alpha = g.fun(x_hat_np) + reg_fun(x_hat_np) + tau_alpha*np.sqrt(N) + N
# Define potential function
loss_fun = lambda x_map : g.fun(x_map) + reg_fun(x_map)


# Compute HPD region

print(
    'f(x_map): ', g.fun(x_hat_np),
    ', g(x_map): ', reg_fun(x_hat_np),
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
    'alpha': alpha,
    'gamma_alpha': gamma_alpha,
    'f_xmap': g.fun(x_hat_np),
    'g_xmap': reg_fun(x_hat_np),
    'h_alpha_N': tau_alpha*np.sqrt(N) + N,
}
save_dic = {
    'x_map': x_hat_np,
    'opt_params': opt_params,
    'hpd_results': hpd_results,
}
saving_path = save_dir + '{:s}_results.npy'.format(save_name)
# We will overwrite the dict with new results
if os.path.isfile(saving_path):
    os.remove(saving_path)
np.save(saving_path, save_dic, allow_pickle=True)


# Plot computed credible region upper bound
upper_bounds = lambda alpha : g.fun(x_hat_np) + reg_fun(x_hat_np) + np.sqrt(16*np.log(3/alpha)) * np.sqrt(N) + N
plot_alphas = np.linspace(0.01, 0.99, 99)

plt.figure()
plt.plot(plot_alphas, upper_bounds(plot_alphas))
plt.title('f(xmap)={:.1f}\ng(xmap)={:.1f}'.format(
    g.fun(x_hat_np), reg_fun(x_hat_np))
)
plt.savefig('{:s}{:s}_gamma.pdf'.format(savefig_dir, save_name))
plt.close()
# plt.show()

# Compute the LCI
superpix_sizes = [30, 20, 10]
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
    x_sol=x_hat_np,
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

    vmin = np.min((x, x_init, x_hat_np))
    vmax = np.max((x, x_init, x_hat_np))
    # err_vmax= 0.6
    cmap='cubehelix'

    plt.figure(figsize=(28,12))
    plt.subplot(241)
    plt.imshow(x, cmap=cmap, vmin=vmin, vmax=vmax);plt.colorbar()
    plt.title('Ground truth')
    plt.subplot(242)
    plt.imshow(x_init, cmap=cmap, vmin=vmin, vmax=vmax);plt.colorbar()
    plt.title('Dirty')
    plt.subplot(243)
    plt.imshow(x_hat_np, cmap=cmap, vmin=vmin, vmax=vmax);plt.colorbar()
    plt.title('MAP estimator')
    plt.subplot(244)
    plt.imshow(x - x_hat_np, cmap=cmap);plt.colorbar()
    plt.title('Ground truth - MAP estimator')
    plt.subplot(245)
    plt.imshow(error_length, cmap=cmap, vmin=0, vmax=0.3);plt.colorbar()
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
    'x_map': x_hat_np,
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

