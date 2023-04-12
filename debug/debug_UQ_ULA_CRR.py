# %%
import os
import numpy as np
from functools import partial
import math
from tqdm import tqdm
import time as time

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


from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.functional import structural_similarity_index_measure 
from torchmetrics.functional import peak_signal_noise_ratio 

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.io as sio
from astropy.io import fits

import large_scale_UQ as luq
from large_scale_UQ.utils import to_numpy, to_tensor
from convex_reg import utils as utils_cvx_reg




# %%
# Optimisation options for the MAP estimation
options = {"tol": 1e-5, "iter": 15000, "update_iter": 4999, "record_iters": False}
# Save param
repo_dir = '/disk/xray0/tl3/repos/large-scale-UQ'
base_savedir = '/disk/xray0/tl3/outputs/large-scale-UQ/sampling/CRR'
save_dir = base_savedir + '/vars/'
savefig_dir = base_savedir + '/figs/'

img_name = 'M31'

# Load img
img_path = '/disk/xray0/tl3/repos/large-scale-UQ/data/imgs/{:s}.fits'.format(img_name)
img_data = fits.open(img_path, memmap=False)

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
op_mask = sio.loadmat(
    repo_dir + '/data/operators_masks/fourier_mask.mat'
)['Ma']

# Matlab's reshape works with 'F'-like ordering
mat_mask = np.reshape(np.sum(op_mask, axis=0), (256,256), order='F').astype(bool)

# Define my torch types
myType = torch.float32
myComplexType = torch.complex64

torch_img = torch.tensor(np.copy(img), dtype=myType, device=device).reshape((1,1) + img.shape)


dim = 256
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


# %%
# Define the likelihood
g = luq.operators.L2Norm_torch(
    sigma=sigma,
    data=torch_y,
    Phi=phi,
)
# Lipschitz constant computed automatically by g, stored in g.beta

# Define real prox
f = luq.operators.RealProx_torch()


# %%
# Load CRR model
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

# L_CRR = model.L.detach().cpu().squeeze().numpy()
# print(f"Lipschitz bound {L_CRR:.3f}")

# [not required] intialize the eigen vector of dimension (size, size) associated to the largest eigen value
model.initializeEigen(size=100)
# compute bound via a power iteration which couples the activations and the convolutions
model.precise_lipschitz_bound(n_iter=100)
# the bound is stored in the model
L_CRR = model.L.data.item()
print(f"Lipschitz bound {L_CRR:.3f}")


# %%

# CRR parameters
reg_params = [250., 1e3, 5e3, 1e4]
mu = 20
# my_lmbda = [1e5] #, 5e4] # [2.5e3, 5e3, 1e4, 2e4, 5e4]


# LCI params
alpha_prob = 0.05

# Compute the MAP-based UQ plots
superpix_MAP_sizes = [32, 16, 8, 4]

# Compute the sampling UQ plots
superpix_sizes = [32,16,8,4,1]

# Sampling alg params
frac_delta = 0.98
frac_burnin = 0.1
n_samples = np.int64(1e4)
thinning = np.int64(1e2)
maxit = np.int64(n_samples * thinning * (1. + frac_burnin))


for it_1 in range(len(reg_params)):

    # Prior parameters
    lmbd = reg_params[it_1]

    # Compute stepsize
    alpha = 0.98 / (g.beta + mu * lmbd * L_CRR)

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
        x_hat = f.prox(x_hat)
        # Positivity constraint
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


    # %%
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

    ### MAP-based UQ

    #function handles to used for ULA
    def _fun(_x, model, mu, lmbd):
        return (lmbd / mu) * model.cost(mu * _x) + g.fun(_x)

    def _grad_fun(_x, g, model, mu, lmbd):
        return  torch.real(g.grad(_x) + lmbd * model(mu * _x))
    
    def _prior_fun(_x, model, mu, lmbd):
        return (lmbd / mu) * model.cost(mu * _x)

    # Evaluation of the potentials
    fun = partial(_fun, model=model, mu=mu, lmbd=lmbd)
    prior_fun = partial(_prior_fun, model=model, mu=mu, lmbd=lmbd)
    # Evaluation of the gradient
    grad_f = partial(_grad_fun, g=g, model=model, mu=mu, lmbd=lmbd)
    # Evaluation of the potential in numpy
    fun_np = lambda _x : fun(luq.utils.to_tensor(_x, dtype=myType)).item()


    # Compute HPD region bound
    N = np_x_hat.size
    tau_alpha = np.sqrt(16*np.log(3/alpha_prob))
    gamma_alpha = fun(x_hat).item() + tau_alpha*np.sqrt(N) + N

    # Compute the LCI
    LCI_iters = 200
    LCI_tol = 1e-4
    LCI_bottom = -10
    LCI_top = 10

    error_p_arr = []
    error_m_arr = []
    computing_time = []

    x_init_np = luq.utils.to_numpy(x_init)

    # Define prefix
    save_MAP_prefix = 'CRR_UQ_MAP_lmbd_{:.1e}'.format(lmbd)


    for superpix_size in superpix_MAP_sizes:

        pr_time_1 = time.process_time()
        wall_time_1 = time.time()

        error_p, error_m, mean = luq.map_uncertainty.create_local_credible_interval(
            x_sol=np_x_hat,
            region_size=superpix_size,
            function=fun_np,
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
            savefig_dir+save_MAP_prefix+'_UQ-MAP_pixel_size_{:d}.pdf'.format(superpix_size)
        )
        plt.close()
        # plt.show()


    print(
        'f(x_map): ', g.fun(x_hat).item(),
        '\ng(x_map): ', prior_fun(x_hat).item(),
        '\ntau_alpha*np.sqrt(N): ', tau_alpha*np.sqrt(N),
        '\nN: ', N,
    )
    print('tau_alpha: ', tau_alpha)
    print('gamma_alpha: ', gamma_alpha.item())
    # 
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
        'f_xmap': g.fun(x_hat).item(),
        'g_xmap': prior_fun(x_hat).item(),
        'h_alpha_N': tau_alpha*np.sqrt(N) + N,
    }
    LCI_params ={
        'iters': LCI_iters,
        'tol': LCI_tol,
        'bottom': LCI_bottom,
        'top': LCI_top,
    }
    save_map_vars = {
        'x_map': np_x_hat,
        'opt_params': opt_params,
        'hpd_results': hpd_results,
        'error_p_arr': error_p_arr,
        'error_m_arr': error_m_arr,
        'computing_time': computing_time,
        'superpix_sizes': superpix_MAP_sizes,
        'LCI_params': LCI_params,
    }
    # We will overwrite the dict with new results
    try:
        saving_map_path = save_dir + save_MAP_prefix + '_MAP_vars.npy'
        if os.path.isfile(saving_map_path):
            os.remove(saving_map_path)
        np.save(saving_map_path, save_map_vars, allow_pickle=True)
    except Exception as e:
        print('Could not save vairables. Exception caught: ', e)    



    # Define saving prefix
    save_prefix = 'ULA_CRR_lmbd_{:.1e}_mu_{:.1e}_nsamples_{:.1e}_thinning_{:.1e}_frac_burn_{:.1e}'.format(
        lmbd, mu, n_samples, thinning, frac_burnin
    )

    ### Sampling

    # Set up sampler
    Lip_total = mu * lmbd * L_CRR + g.beta 

    #step size for ULA
    delta = frac_delta / Lip_total

    # Sampling alg params
    burnin = np.int64(n_samples * thinning * frac_burnin)
    X = x_init.clone()
    MC_X = np.zeros((n_samples, X.shape[1], X.shape[2]))
    logpi_thinning_trace = np.zeros((n_samples, 1))
    thinned_trace_counter = 0

    nrmse_values = []
    psnr_values = []
    ssim_values = []
    logpi_eval = []

    # %%
    start_time = time.time()
    for i_x in tqdm(range(maxit)):

        # Update X
        X = luq.sampling.ULA_kernel(X, delta, grad_f)

        if i_x == burnin:
            # Initialise recording of sample summary statistics after burnin period
            post_meanvar = luq.utils.welford(X)
            absfouriercoeff = luq.utils.welford(torch.fft.fft2(X).abs())
        elif i_x > burnin:
            # update the sample summary statistics
            post_meanvar.update(X)
            absfouriercoeff.update(torch.fft.fft2(X).abs())

            # collect quality measurements
            current_mean = post_meanvar.get_mean()
            psnr_values.append(peak_signal_noise_ratio(torch_img, current_mean).item())
            ssim_values.append(structural_similarity_index_measure(torch_img, current_mean).item())
            # [TL] Need to use pytorch version of NRMSE!
            nrmse_values.append(luq.functions.measures.NRMSE(torch_img, current_mean))
            logpi_eval.append(fun(X).item())

            # collect thinned trace
            if np.mod(i_x - burnin, thinning) == 0:
                MC_X[thinned_trace_counter] = X.detach().cpu().numpy()
                logpi_thinning_trace[thinned_trace_counter] = fun(X).item()
                thinned_trace_counter += 1

    end_time = time.time()
    elapsed = end_time - start_time    

    current_mean = post_meanvar.get_mean()
    current_var = post_meanvar.get_var().detach().cpu().squeeze()


    # %%
    cmap = 'cubehelix'

    quantiles, st_dev_down, means_list = luq.map_uncertainty.compute_UQ(
        MC_X, superpix_sizes, alpha_prob
    )

    for it_4, pix_size in enumerate(superpix_sizes):

        # Plot UQ
        fig = plt.figure(figsize=(20,5))

        plt.subplot(131)
        ax = plt.gca()
        ax.set_title(f'Mean value, <Mean val>={np.mean(means_list[it_4]):.2e} pix size={pix_size:d}')
        im = ax.imshow(means_list[it_4], cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.set_yticks([]);ax.set_xticks([])

        plt.subplot(132)
        ax = plt.gca()
        ax.set_title(f'St Dev, <St Dev>={np.mean(st_dev_down[it_4]):.2e} pix size={pix_size:d}')
        im = ax.imshow(st_dev_down[it_4], cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.set_yticks([]);ax.set_xticks([])

        plt.subplot(133)
        LCI = quantiles[it_4][1,:,:] - quantiles[it_4][0,:,:]
        ax = plt.gca()
        ax.set_title(f'LCI, <LCI>={np.mean(LCI):.2e} pix size={pix_size:d}')
        im = ax.imshow(LCI, cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.set_yticks([]);ax.set_xticks([])
        plt.savefig(savefig_dir+save_prefix+'_UQ_pixel_size_{:d}.pdf'.format(pix_size))
        plt.close()



    params = {
        'maxit': maxit,
        'n_samples': n_samples,
        'thinning': thinning,
        'frac_burnin': frac_burnin,
        'delta': delta,
        'frac_delta': frac_delta,
        'mu': mu,
        'lmbd': lmbd,
        'superpix_sizes': np.array(superpix_sizes),
        'alpha_prob': alpha_prob,
    }
    save_vars = {
        'X_ground_truth': torch_img.detach().cpu().squeeze().numpy(),
        'X_dirty': x_init.detach().cpu().squeeze().numpy(),
        'X_MAP': np_x_hat,
        'X_MMSE': np.mean(MC_X, axis=0),
        'post_meanvar': post_meanvar,
        'absfouriercoeff': absfouriercoeff,
        'MC_X': MC_X,
        'logpi_thinning_trace': logpi_thinning_trace,
        'X': to_numpy(X),
        'quantiles': quantiles,
        'st_dev_down': st_dev_down,
        'means_list': means_list,
        'params': params,
        'elapsed_time': elapsed,
    }


    # %%
    # Plot

    luq.utils.plot_summaries(
        x_ground_truth=torch_img.detach().cpu().squeeze().numpy(),
        x_dirty=x_init.detach().cpu().squeeze().numpy(),
        post_meanvar=post_meanvar,
        post_meanvar_absfourier=absfouriercoeff,
        cmap=cmap,
        save_path=savefig_dir+save_prefix+'_summary_plots.pdf'
    )


    fig, ax = plt.subplots()
    ax.set_title("log pi")
    ax.plot(np.arange(1,len(logpi_eval)+1), logpi_eval)
    plt.savefig(savefig_dir+save_prefix+'_log_pi_sampling.pdf')
    # plt.show()
    plt.close()

    fig, ax = plt.subplots()
    ax.set_title("log pi thinning")
    ax.plot(np.arange(1,len(logpi_thinning_trace[:-1])+1), logpi_thinning_trace[:-1])
    plt.savefig(savefig_dir+save_prefix+'_log_pi_thinning_sampling.pdf')
    # plt.show()
    plt.close()


    MC_X_mean = np.mean(MC_X, axis=0)

    fig, ax = plt.subplots()
    ax.set_title(f"Image MMSE (Regularization Cost {model.cost(mu*to_tensor(MC_X_mean, device=device))[0].item():.1f}, PSNR: {peak_signal_noise_ratio(to_tensor(MC_X_mean, device=device), torch_img).item():.2f})")
    im = ax.imshow(MC_X_mean, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.savefig(savefig_dir+save_prefix+'_MMSE_sampling.pdf')
    # plt.show()
    plt.close()

    fig, ax = plt.subplots()
    ax.set_title(f"Image VAR")
    im = ax.imshow(current_var, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_yticks([]);ax.set_xticks([])
    plt.savefig(savefig_dir+save_prefix+'_variance_sampling.pdf')
    # plt.show()
    plt.close()

    nLags = 100
    if nLags < n_samples:
        luq.utils.autocor_plots(
            MC_X,
            current_var,
            "ULA",
            nLags=nLags,
            save_path=savefig_dir+save_prefix+'_autocorr_plot.pdf'
        )


    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    ax[0].set_title("NRMSE")
    ax[0].plot(np.arange(1, len(nrmse_values) + 1), nrmse_values)

    ax[1].set_title("SSIM")
    ax[1].plot(np.arange(1, len(ssim_values) + 1), ssim_values)

    ax[2].set_title("PSNR")
    ax[2].plot(np.arange(1, len(psnr_values) + 1), psnr_values)

    plt.savefig(savefig_dir+save_prefix+'_NRMSE_SSIM_PSNR_evolution.pdf')
    plt.close()

    # Save variables
    try:
        save_path = '{:s}{:s}{:s}'.format(
            save_dir, save_prefix, '_vars.npy'
        )
        np.save(save_path, save_vars, allow_pickle=True)
    except Exception as e:
        print('Could not save vairables. Exception caught: ', e)

