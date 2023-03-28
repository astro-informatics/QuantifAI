# %%
import os
import numpy as np

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


from functools import partial
import math
import time as time

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.functional import structural_similarity_index_measure 
from torchmetrics.functional import peak_signal_noise_ratio 

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
sns.set(font_scale=1.5)
# plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"

import scipy.io as sio
from astropy.io import fits

import large_scale_UQ as luq
from large_scale_UQ.utils import to_numpy, to_tensor
from convex_reg import utils as utils_cvx_reg



# %%


# %%
# Optimisation options for the MAP estimation
options = {"tol": 1e-5, "iter": 5000, "update_iter": 4999, "record_iters": False}


# %%
# Save param
repo_dir = '/disk/xray0/tl3/repos/large-scale-UQ'
# repo_dir = '/Users/tl/Documents/research/repos/proj-convex-UQ/large-scale-UQ'
save_dir = repo_dir + '/debug/torch_output_reg_strength/outputs/'
savefig_dir = repo_dir + '/debug/torch_output_reg_strength/figs/'



# %%
img_name = 'M31'

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


# %%
# %%
# Load op from X Cai
mask_path = repo_dir + '/data/operators_masks/fourier_mask.mat'
op_mask = sio.loadmat(mask_path)['Ma']

# Matlab's reshape works with 'F'-like ordering
mat_mask = np.reshape(np.sum(op_mask, axis=0), (256,256), order='F').astype(bool)

# %%

torch_img = torch.tensor(np.copy(img), dtype=torch.float32, device=device).reshape((1,1) + img.shape)

# %%
dim = 256
phi = luq.operators.MaskedFourier_torch(
    dim=dim, 
    ratio=0.5 ,
    mask=mat_mask,
    norm='ortho',
    device=device
)



# %%
# Define X Cai noise level
sigma = 0.0024

y = phi.dir_op(torch_img).detach().cpu().squeeze().numpy()

# Generate noise
rng = np.random.default_rng(seed=0)
n = rng.normal(0, sigma, y[y!=0].shape)
# Add noise
y[y!=0] += n

# Observation
torch_y = torch.tensor(np.copy(y), device=device, dtype=torch.complex64).reshape((1,) + img.shape)
x_init = torch.abs(phi.adj_op(torch_y))



# %%
# Define the likelihood
g = luq.operators.L2Norm_torch(
    sigma=sigma,
    data=torch_y,
    Phi=phi,
)
g.beta = 1.0 / sigma ** 2



# %%
torch.set_grad_enabled(False)
torch.set_num_threads(4)

sigma_training = 5
t_model = 5
dir_name = '/disk/xray0/tl3/repos/convex_ridge_regularizers/trained_models/'
exp_name = f'Sigma_{sigma_training}_t_{t_model}/'
model = utils_cvx_reg.load_model(dir_name+exp_name, 'cuda:0', device_type='gpu')

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


# %%
lmbd_list = np.logspace(start=np.log10(1e2), stop=np.log10(1e6), num=25)
lmbd_list


x_hat_list = []
x_hat_np_list = []
gamma_alpha_list = []
prior_list = []
likelihood_list = []
const_gamma_alpha_list = []
psnr_map_list = []

error_p_list = []
error_m_list = []

# LCI parameters
superpix_sizes = [32, 16, 8]
LCI_iters = 200
LCI_tol = 1e-5
LCI_bottom = -10
LCI_top = 10

alpha = 0.05

zoom_id = 12
prefix = 'UQ_MAP_CRR_reg_strength'

mean_LCI = np.zeros((len(superpix_sizes), len(lmbd_list)))
computing_time = np.zeros((len(superpix_sizes), len(lmbd_list)))

for it_lmbd in range(len(lmbd_list)):

    # Prior parameters
    lmbd = lmbd_list[it_lmbd]
    mu = 20

    # Compute stepsize
    alpha = 1. / ( 1. + g.beta + mu * lmbd * L)

    # initialization
    x_hat = torch.clone(x_init)
    z = torch.clone(x_init)
    t = 1


    for it in range(options['iter']):
        x_hat_old = torch.clone(x_hat)
        # grad = g.grad(z.squeeze()) +  lmbd * model(mu * z)
        x_hat = z - alpha *(
            g.grad(z) + lmbd * model(mu * z)
        )
        # Reality constraint
        x_hat =  torch.real(x_hat)

        t_old = t 
        t = 0.5 * (1 + math.sqrt(1 + 4*t**2))
        z = x_hat + (t_old - 1)/t * (x_hat - x_hat_old)

        # relative change of norm for terminating
        res = (torch.norm(x_hat_old - x_hat)/torch.norm(x_hat_old)).item()

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

    x_hat_list.append(x_hat)
    x_hat_np_list.append(luq.utils.to_numpy(x_hat))

    psnr_map = psnr(ground_truth, x_hat_np_list[it_lmbd], data_range=ground_truth.max()-ground_truth.min())
    psnr_map_list.append(psnr_map)

    N = x_hat_np_list[it_lmbd].size
    tau_alpha = np.sqrt(16*np.log(3/alpha))

    # _reg_fun = lambda h, _x: h.fun(h.dir_op(_x))
    # def _reg_fun(_x, h):
    #     return h.fun(h.dir_op(_x))
    # reg_fun = partial(_reg_fun, h=h)

    reg_fun = lambda _x : (lmbd / mu) * model.cost(mu * _x)

    def _fun(_x, model, mu, lmbd):
        return (lmbd / mu) * model.cost(mu * _x) + g.fun(_x)

    fun = partial(_fun, model=model, mu=mu, lmbd=lmbd)

    loss_fun_torch = lambda _x : fun(_x)
    loss_fun_np = lambda _x : fun(luq.utils.to_tensor(_x, dtype=torch.float32)).item()
    # loss_fun = lambda _x : g.fun(_x)

    gamma_alpha_list.append(loss_fun_torch(x_hat).item() + tau_alpha*np.sqrt(N) + N)
    prior_list.append(reg_fun(x_hat).item())
    likelihood_list.append(g.fun(x_hat).item())
    const_gamma_alpha_list.append(tau_alpha*np.sqrt(N) + N)



    # Compute the LCI
    error_p_arr = []
    error_m_arr = []

    x_init_np = luq.utils.to_numpy(x_init)

    for it_pix, superpix_size in enumerate(superpix_sizes):

        pr_time_1 = time.process_time()
        wall_time_1 = time.time()

        error_p, error_m, mean = luq.map_uncertainty.create_local_credible_interval(
        x_sol=x_hat_np_list[it_lmbd],
        region_size=superpix_size,
        function=loss_fun_np,
        bound=gamma_alpha_list[it_lmbd],
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
        computing_time[it_pix, it_lmbd] = wall_time_2 - wall_time_1

        mean_LCI[it_pix, it_lmbd] = np.mean(error_length)
    
    error_p_list.append(error_p_arr)
    error_m_list.append(error_m_arr)

params = {
    'lmbd_list': lmbd_list,
    'mu': mu,
    'superpix_sizes': np.array(superpix_size),
    'LCI_iters': LCI_iters,
    'LCI_tol': LCI_tol,
    'LCI_bottom': LCI_bottom,
    'LCI_top': LCI_top,
    'alpha': alpha,
    'otpim_options': options,
    'sigma_training': sigma_training,
    't_model': t_model,
    'sigma_noise': sigma,
}

save_vars = {
    'X_MAPs': np.array(x_hat_np_list),
    'mean_LCI': mean_LCI,
    'computing_LCI_time': computing_time,
    'gamma_alpha_list': np.array(gamma_alpha_list),
    'prior_list': np.array(prior_list),
    'likelihood_list': np.array(likelihood_list),
    'const_gamma_alpha_list': np.array(const_gamma_alpha_list),
    'error_p_list': np.array(error_p_list, dtype=object),
    'error_m_list': np.array(error_m_list, dtype=object),
    'params': params,
}


save_path = '{:s}{:s}{:s}'.format(save_dir, prefix, '_vars.npy')
np.save(save_path, save_vars, allow_pickle=True)


kwargs = dict(linewidth=2, linestyle='dashed', markersize=6, marker='^', alpha=0.5)
def_cmap = plt.get_cmap("tab10")

plt.figure(figsize=(10,12))
axs = plt.gca()
axs.plot(lmbd_list, np.array(likelihood_list), '-o', label='f(x_map)')
axs.plot(lmbd_list, np.array(prior_list), '-o', label='g(x_map)')
axs.plot(lmbd_list, np.array(const_gamma_alpha_list), '-o', label='tau_alpha')
axs.legend(fontsize=18, loc='upper center')
axs.set_ylabel(r'Potentials')
axs.set_xlabel(r'Reg strength')
ax2 = axs.twinx()
ax2.plot(lmbd_list, np.array(psnr_map_list), color=def_cmap(3), **kwargs)
ax2.set_ylabel(r'PSNR(x_MAP)')
ax2.grid(False)
plt.savefig('{:s}{:s}{:s}'.format(savefig_dir, prefix, '_potentials_plot.pdf'))
plt.close()

plt.figure(figsize=(10,12))
axs = plt.gca()
axs.plot(lmbd_list[:12], np.array(likelihood_list)[:zoom_id], '-o', label='f(x_map)')
axs.plot(lmbd_list[:12], np.array(prior_list)[:zoom_id], '-o', label='g(x_map)')
axs.plot(lmbd_list[:12], np.array(const_gamma_alpha_list)[:zoom_id], '-o', label='tau_alpha')
axs.legend(fontsize=18, loc='upper center')
axs.set_ylabel(r'Potentials')
axs.set_xlabel(r'Reg strength')
ax2 = axs.twinx()
ax2.plot(lmbd_list, np.array(psnr_map_list)[:zoom_id], color=def_cmap(3), **kwargs)
ax2.set_ylabel(r'PSNR(x_MAP)')
ax2.grid(False)
plt.savefig('{:s}{:s}{:s}'.format(savefig_dir, prefix, '_potentials_plot_zoom.pdf'))
plt.close()


kwargs = dict(linewidth=2, linestyle='dashed', markersize=6, marker='^', alpha=0.5)
def_cmap = plt.get_cmap("tab10")

plt.figure(figsize=(10,12))
axs = plt.gca()
for it in range(len(superpix_sizes)):
    axs.plot(lmbd_list, mean_LCI[it,:], '-o', label='Pix size %d'%(superpix_sizes[it]))
axs.legend(fontsize=18, loc='upper center')
axs.set_ylabel(r'Potentials')
axs.set_xlabel(r'Reg strength')
ax2 = axs.twinx()
ax2.plot(lmbd_list, np.array(psnr_map_list), color=def_cmap(3), **kwargs)
ax2.set_ylabel(r'PSNR(x_MAP)')
ax2.grid(False)
plt.savefig('{:s}{:s}{:s}'.format(savefig_dir, prefix, '_LCI_mean_plot.pdf'))
plt.close()
# plt.show()

kwargs = dict(linewidth=2, linestyle='dashed', markersize=6, marker='^', alpha=0.5)
def_cmap = plt.get_cmap("tab10")

plt.figure(figsize=(10,12))
axs = plt.gca()
for it in range(len(superpix_sizes)):
    axs.plot(lmbd_list, mean_LCI[it,:zoom_id], '-o', label='Pix size %d'%(superpix_sizes[it]))
axs.legend(fontsize=18, loc='upper center')
axs.set_ylabel(r'Potentials')
axs.set_xlabel(r'Reg strength')
ax2 = axs.twinx()
ax2.plot(lmbd_list, np.array(psnr_map_list)[:zoom_id], color=def_cmap(3), **kwargs)
ax2.set_ylabel(r'PSNR(x_MAP)')
ax2.grid(False)
plt.savefig('{:s}{:s}{:s}'.format(savefig_dir, prefix, '_LCI_mean_plot_zoom.pdf'))
plt.close()
# plt.show()



kwargs = dict(linewidth=2, linestyle='dashed', markersize=6, marker='^', alpha=0.5)
def_cmap = plt.get_cmap("tab10")

fig, axs = plt.subplots(2,2, figsize=(24,16))

for it in range(len(superpix_sizes)):
    axs[0,0].plot(lmbd_list, mean_LCI[it,:], '-o', label='Pix size %d'%(superpix_sizes[it]))
axs[0,0].legend(fontsize=18)
axs[0,0].set_ylabel(r'<LCI>')
axs[0,0].set_xlabel(r'Reg strength')
ax2 = axs[0,0].twinx()
ax2.plot(lmbd_list, np.array(psnr_map_list), color=def_cmap(3), **kwargs)
ax2.set_ylabel(r'PSNR(x_MAP)')
ax2.grid(False)

# 
axs[1,0].plot(lmbd_list, np.array(likelihood_list), '-o', label='likelihood(x_map)')
axs[1,0].plot(lmbd_list, np.array(prior_list), '-o', label='prior(x_map)')
axs[1,0].plot(lmbd_list, np.array(const_gamma_alpha_list), '-o', label='tau_alpha')
axs[1,0].legend(fontsize=18, loc='upper center')
axs[1,0].set_ylabel(r'Potentials')
axs[1,0].set_xlabel(r'Reg strength')
ax2 = axs[1,0].twinx()
ax2.plot(lmbd_list, np.array(psnr_map_list), color=def_cmap(3), **kwargs)
ax2.set_ylabel(r'PSNR(x_MAP)')
ax2.grid(False)

for it in range(len(superpix_sizes)):
    axs[0,1].plot(lmbd_list, mean_LCI[it,:zoom_id], '-o', label='Pix size %d'%(superpix_sizes[it]))
axs[0,1].legend(fontsize=18)
axs[0,1].set_ylabel(r'<LCI>')
axs[0,1].set_xlabel(r'Reg strength')
ax2 = axs[0,1].twinx()
ax2.plot(lmbd_list[:zoom_id], np.array(psnr_map_list)[:zoom_id], color=def_cmap(3), **kwargs)
ax2.set_ylabel(r'PSNR(x_MAP)')
ax2.grid(False)
# 

axs[1,1].plot(lmbd_list, np.array(likelihood_list)[:zoom_id], '-o', label='likelihood(x_map)')
axs[1,1].plot(lmbd_list, np.array(prior_list)[:zoom_id], '-o', label='prior(x_map)')
axs[1,1].plot(lmbd_list, np.array(const_gamma_alpha_list)[:zoom_id], '-o', label='tau_alpha')
axs[1,1].legend(fontsize=18, loc='center right')
axs[1,1].set_ylabel(r'Potentials')
axs[1,1].set_xlabel(r'Reg strength')
ax2 = axs[1,1].twinx()
ax2.plot(lmbd_list[:zoom_id], np.array(psnr_map_list)[:zoom_id], color=def_cmap(3), **kwargs)
ax2.set_ylabel(r'PSNR(x_MAP)')
ax2.grid(False)
plt.tight_layout()
save_path = '{:s}{:s}{:s}'.format(savefig_dir, prefix, '_reg_strength_plot.pdf')
plt.savefig(save_path)
plt.close()
# plt.show()




