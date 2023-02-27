
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
options = {"tol": 1e-4, "iter": 5000, "update_iter": 50, "record_iters": False}
save_dir = './debug/output/'
savefig_dir = './debug/figs/'
# optimization settings
wav =  ["db8"]# ["db1", "db4"]                                     # Wavelet dictionaries to combine
levels = 4 # 3                                               # Wavelet levels to consider [1-6]
reg_param = 1.e1
img_name = 'M31'
save_name = '{:s}_256_wavelet-{:s}_{:d}_reg_{:.1f}'.format(
    img_name, wav[0], levels, reg_param
)


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



## Primal-dual FB Wavelet-based denoiser

# Define the grad
g = optpr.grad_operators.l2_norm(sigma, y, phi)
g.beta = 1.0 / sigma ** 2

# Define the wavelet dict
psi = optpr.linear_operators.dictionary(wav, levels, x.shape)  # Wavelet linear operator

# Define the l1 norm prox with the dict psi
l1_reg = np.max(np.abs(psi.dir_op(abs(phi.adj_op(y))))) * reg_param
print('l1_reg: ', l1_reg)
h = optpr.prox_operators.l1_norm(l1_reg, psi)
h.beta = 1.0

# Real prox 
r = optpr.prox_operators.real_prox()
# f = None

# Run the PD algorithm
wvlt_best_estimate, wvlt_diagnostics = optpr.primal_dual.FBPD(
    x_init=x_init, options=options, g=g, f=None, h=h, r=r
)

# Plot results
images = [x, x_init, np.abs(wvlt_best_estimate), x-np.abs(wvlt_best_estimate)]
labels = ["Truth", "Dirty", "Reconstruction", "Residual (x - x^hat)"]

fig, axs = plt.subplots(1,4, figsize=(24,6), dpi=200)
for i in range(4):
    im = axs[i].imshow(images[i], cmap='cubehelix', vmax=np.nanmax(images[i]), vmin=np.nanmin(images[i]))
    divider = make_axes_locatable(axs[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    if i > 0:   
        stats_str = '\n(PSNR: {},\n SNR: {}, SSIM: {})'.format(
            round(psnr(ground_truth, images[i],data_range=ground_truth.max()-ground_truth.min()), 2),
            round(eval_snr(x, images[i]), 2),
            round(ssim(ground_truth, images[i]), 2),
            )
        labels[i] += stats_str
        print(labels[i])
    axs[i].set_title(labels[i], fontsize=16)
    axs[i].axis('off')
plt.savefig('{:s}{:s}_MAP.pdf'.format(savefig_dir, save_name))
plt.close()
# plt.show()


# Compute HPD region

x_hat_np = np.abs(wvlt_best_estimate)

alpha = 0.01
N = x_hat_np.size
tau_alpha = np.sqrt(16*np.log(3/alpha))

# _reg_fun = lambda h, _x: h.fun(h.dir_op(_x))
def _reg_fun(_x, h):
    return h.fun(h.dir_op(_x))
reg_fun = partial(_reg_fun, h=h)

loss_fun = lambda _x : g.fun(_x) +  reg_fun(_x)
# loss_fun = lambda _x : g.fun(_x)

gamma_alpha = loss_fun(x_hat_np) + tau_alpha*np.sqrt(N) + N


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
    'wav': wav,
    'levels': levels,
    'reg_param': reg_param,
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
    plt.imshow(mean, cmap=cmap);plt.colorbar();plt.title('Mean')
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

