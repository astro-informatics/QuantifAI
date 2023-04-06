
import sys
import os
import time
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
# plt.style.use('dark_background')
# plt.rcParams["font.family"] = "serif"



# Parameters
options = {"tol": 1e-5, "iter": 5000, "update_iter": 50, "record_iters": False}

repo_dir = '/disk/xray0/tl3/repos/large-scale-UQ'
save_dir = repo_dir + '/debug/output/'
savefig_dir = repo_dir + '/debug/figs/'

# optimization settings
wavs =  ["db8"]# ["db1", "db4"]                                     # Wavelet dictionaries to combine
levels = 4 # 3                                               # Wavelet levels to consider [1-6]
reg_param = 2.e-3
img_name = 'M31'

# Saving names
save_name = '{:s}_256_wavelet-{:s}_{:d}_reg_{:.1f}'.format(
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


# Define the wavelet dict
# Define the l1 norm with dict psi
# gamma = torch.max(torch.abs(psi.dir_op(y_torch))) * reg_param
psi = luq.operators.DictionaryWv_torch(wavs, levels)

h = luq.operators.L1Norm_torch(1., psi, op_to_coeffs=True)
gamma = h._get_max_abs_coeffs(h.dir_op(torch.clone(x_init))) * reg_param
h.gamma = gamma
h.beta = 1.0



# Compute stepsize
alpha = 1. / (1. + g.beta)

# Run the optimisation
x_hat, diagnostics = luq.optim.FB_torch(
    x_init,
    options=options,
    g=g,
    f=f,
    h=h,
    alpha=alpha,
    tau=1.,
    viewer=None
)


np_x_init = to_numpy(x_init)
np_x = np.copy(x)
np_x_hat = to_numpy(x_hat)



# %%
images = [np_x, np_x_init, np_x_hat, np_x - np.abs(np_x_hat)]
labels = ["Truth", "Dirty", "Reconstruction", "Residual (x - x^hat)"]
fig, axs = plt.subplots(1,4, figsize=(24,6), dpi=200)
for i in range(4):
    im = axs[i].imshow(images[i], cmap='cubehelix', vmax=np.nanmax(images[i]), vmin=np.nanmin(images[i]))
    divider = make_axes_locatable(axs[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    if i > 0:   
        stats_str = '\n(PSNR: {},\n SNR: {}, SSIM: {})'.format(
            round(psnr(ground_truth, images[i], data_range=ground_truth.max()-ground_truth.min()), 2),
            round(luq.utils.eval_snr(x, images[i]), 2),
            round(ssim(ground_truth, images[i], data_range=ground_truth.max()-ground_truth.min()), 2),
            )
        labels[i] += stats_str
        print(labels[i])
    axs[i].set_title(labels[i], fontsize=16)
    axs[i].axis('off')
plt.savefig('{:s}{:s}_MYULA_wavelets_reg_param_{:.1e}_optim_MAP.pdf'.format(savefig_dir, img_name, reg_param))
plt.close()


# Compute HPD region

use_CAI_solution = False

if use_CAI_solution:
    matlab_output = '/Users/tliaudat/Documents/postdoc/github/large-scale-UQ/debug/matlab_output/modified_Cai_results.mat'
    cai_resutls = sio.loadmat(matlab_output)
    np_x_hat = cai_resutls['sol_ana']
    save_name = save_name + '_Cai_sol'


# Compute HPD region
alpha = 0.05

N = np_x_hat.size
tau_alpha = np.sqrt(16*np.log(3/alpha))

# _reg_fun = lambda h, _x: h.fun(h.dir_op(_x))
def _reg_fun(_x, h):
    return h.fun(h.dir_op(_x))
reg_fun = partial(_reg_fun, h=h)

loss_fun_torch = lambda _x : g.fun(_x) +  reg_fun(_x)
loss_fun_np = lambda _x : g.fun(
    luq.utils.to_tensor(_x, dtype=torch.float64)
).item() +  reg_fun(luq.utils.to_tensor(_x, dtype=torch.float64)).item()
# loss_fun = lambda _x : g.fun(_x)

gamma_alpha = loss_fun_torch(x_hat).item() + tau_alpha*np.sqrt(N) + N


print(
    'f(x_map): ', g.fun(x_hat).item(),
    '\ng(x_map): ', reg_fun(x_hat).item(),
    '\ntau_alpha*np.sqrt(N): ', tau_alpha*np.sqrt(N),
    '\nN: ', N,
)
print('tau_alpha: ', tau_alpha)
print('gamma_alpha: ', gamma_alpha.item())



opt_params = {
    'wav': wavs,
    'levels': levels,
    'reg_param': reg_param,
    'sigma_noise': sigma,
    'opt_tol': options['tol'],
    'opt_max_iter': options['iter'],
}
hpd_results = {
    'alpha': alpha,
    'gamma_alpha': gamma_alpha,
    'f_xmap': g.fun(x_hat).item(),
    'g_xmap': reg_fun(x_hat).item(),
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
likelihood_prior_map_np = g.fun(x_hat).item() + reg_fun(x_hat).item()
upper_bounds = lambda alpha : likelihood_prior_map_np + np.sqrt(16*np.log(3/alpha)) * np.sqrt(N) + N
plot_alphas = np.linspace(0.01, 0.99, 99)

plt.figure()
plt.plot(plot_alphas, upper_bounds(plot_alphas))
plt.title('f(xmap)={:.1f}\ng(xmap)={:.1f}'.format(
    g.fun(x_hat).item(), reg_fun(x_hat).item())
)
plt.savefig('{:s}{:s}_gamma.pdf'.format(savefig_dir, save_name))
plt.close()
# plt.show()


# Compute the LCI
superpix_sizes = [32, 16, 8]
LCI_iters = 200
LCI_tol = 1e-4
LCI_bottom = -10
LCI_top = 10

error_p_arr = []
error_m_arr = []
computing_time = []

x_init_np = luq.utils.to_numpy(x_init)

for superpix_size in superpix_sizes:

    pr_time_1 = time.process_time()
    wall_time_1 = time.time()

    error_p, error_m, mean = luq.map_uncertainty.create_local_credible_interval(
        x_sol=np_x_hat,
        region_size=superpix_size,
        function=loss_fun_np,
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

    vmin = np.min((x, x_init_np, np_x_hat))
    vmax = np.max((x, x_init_np, np_x_hat))
    # err_vmax= 0.6
    cmap='cubehelix'

    plt.figure(figsize=(28,12))
    plt.subplot(241)
    plt.imshow(x, cmap=cmap, vmin=vmin, vmax=vmax);plt.colorbar()
    plt.title('Ground truth')
    plt.subplot(242)
    plt.imshow(x_init_np, cmap=cmap, vmin=vmin, vmax=vmax);plt.colorbar()
    plt.title('Dirty')
    plt.subplot(243)
    plt.imshow(np_x_hat, cmap=cmap, vmin=vmin, vmax=vmax);plt.colorbar()
    plt.title('MAP estimator')
    plt.subplot(244)
    plt.imshow(x - np_x_hat, cmap=cmap);plt.colorbar()
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

