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
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
options = {"tol": 1e-5, "iter": 14000, "update_iter": 4999, "record_iters": False}
# Save param
repo_dir = '/disk/xray0/tl3/repos/large-scale-UQ'
base_savedir = '/disk/xray0/tl3/outputs/large-scale-UQ/sampling/wavelets'
save_dir = base_savedir + '/vars/'
savefig_dir = base_savedir + '/figs/'

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
# Load op from X Cai
op_mask = sio.loadmat(
    repo_dir + '/data/operators_masks/fourier_mask.mat'
)['Ma']

# Matlab's reshape works with 'F'-like ordering
mat_mask = np.reshape(np.sum(op_mask, axis=0), (256,256), order='F').astype(bool)

# Define my torch types
myType = torch.float64
myComplexType = torch.complex128

torch_img = torch.tensor(np.copy(img), dtype=myType, device=device).reshape((1,1) + img.shape)

# %%
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

# Iterate over
my_frac_delta = [0.98]
reg_params = [5., 10., 20., 40.]

# Wavelet parameters
wavs_list = ['db8']
levels = 4

# LCI params
alpha_prob = 0.05

# Sampling alg params
frac_burnin = 0.1
n_samples = np.int64(1e4)
thinning = np.int64(1e2)
maxit = np.int64(n_samples * thinning * (1. + frac_burnin))



for it_param, reg_param in enumerate(reg_params):

    # Define the wavelet dict
    # Define the l1 norm with dict psi
    psi = luq.operators.DictionaryWv_torch(wavs_list, levels)

    h = luq.operators.L1Norm_torch(1., psi, op_to_coeffs=True)
    # gamma = h._get_max_abs_coeffs(h.dir_op(torch.clone(x_init))) * reg_param
    # h.gamma = gamma
    h.gamma = reg_param

    # Compute stepsize
    alpha = 0.98 / g.beta

    # Effective threshold
    print('Threshold: ', h.gamma * alpha)

    # Run the optimisation
    x_hat, diagnostics = luq.optim.FB_torch(
        x_init,
        options=options,
        g=g,
        f=f,
        h=h,
        alpha=alpha,
        tau=alpha,
        viewer=None
    )


    # %%
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


    ### MAP-based UQ

    # Define prior potential
    fun_prior = lambda _x : h._fun_coeffs(h.dir_op(_x))
    # Define posterior potential
    loss_fun_torch = lambda _x : g.fun(_x) +  fun_prior(_x)
    # Numpy version of the posterior potential
    loss_fun_np = lambda _x : g.fun(
        luq.utils.to_tensor(_x, dtype=myType)
    ).item() +  fun_prior(luq.utils.to_tensor(_x, dtype=myType)).item()

    # Compute HPD region bound
    N = np_x_hat.size
    tau_alpha = np.sqrt(16*np.log(3/alpha_prob))
    gamma_alpha = loss_fun_torch(x_hat).item() + tau_alpha*np.sqrt(N) + N


    # Compute the LCI
    superpix_sizes = [32, 16, 8, 4]
    LCI_iters = 200
    LCI_tol = 1e-4
    LCI_bottom = -10
    LCI_top = 10

    error_p_arr = []
    error_m_arr = []
    computing_time = []

    x_init_np = luq.utils.to_numpy(x_init)

    # Define prefix
    save_MAP_prefix = 'wavelets_UQ_MAP_reg_param_{:.1e}'.format(reg_param)

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
        plt.imshow(error_length, cmap=cmap);plt.colorbar()
        plt.title('LCI (max={:.5f})\n (mean={:.5f})'.format(
            np.max(error_length), np.mean(error_length))
        )
        plt.subplot(246)
        plt.imshow(error_length - np.mean(error_length), cmap='viridis');plt.colorbar()
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
        '\ng(x_map): ', fun_prior(x_hat).item(),
        '\ntau_alpha*np.sqrt(N): ', tau_alpha*np.sqrt(N),
        '\nN: ', N,
    )
    print('tau_alpha: ', tau_alpha)
    print('gamma_alpha: ', gamma_alpha.item())
    # 
    opt_params = {
        'wav': wavs_list,
        'levels': levels,
        'reg_param': reg_param,
        'sigma_noise': sigma,
        'opt_tol': options['tol'],
        'opt_max_iter': options['iter'],
    }
    hpd_results = {
        'alpha': alpha_prob,
        'gamma_alpha': gamma_alpha,
        'f_xmap': g.fun(x_hat).item(),
        'g_xmap': fun_prior(x_hat).item(),
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
        'superpix_sizes': superpix_sizes,
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


    ### Sampling

    # Define sampling parameters
    L_likelihood = g.beta
    # Define MY lambda parameter
    lmbd = 1 / L_likelihood
    # Compute MY envelope Lipschitz param
    L_g = 1 / lmbd
    # Total Lipschitz
    L = L_g + L_likelihood
    # Define sampling's step size
    frac_delta = 0.98
    delta = frac_delta / L

    # Change model's parameter [optional]
    reg_param_sampling = reg_param
    h.gamma = reg_param_sampling


    print('delta', delta)
    print('lmbd: ', lmbd)
    print('prox thresh: ', h.gamma*lmbd)
    print('(1. - (delta / lmbd)) ', (1. - (delta/lmbd)))


    # Function handles to used for ULA
    # Define likelihood functions
    fun_likelihood = lambda _x : g.fun(_x)
    grad_likelihood = lambda _x : g.grad(_x)
    # Define prior potential
    fun_prior = lambda _x : h._fun_coeffs(h.dir_op(_x))
    # Define prior evaluation
    sub_op = lambda _x1, _x2 : _x1 - _x2
    prox_prior_cai = lambda _x, lmbd : torch.clone(_x) + h.adj_op(h._op_to_two_coeffs(
        h.prox(h.dir_op(_x), lmbd),
        h.dir_op(_x), sub_op
    ))
    # Define posterior potential
    logPi = lambda _z :  fun_likelihood(_z) + fun_prior(_z)
    # Define reality prox
    real_prox = lambda _z : torch.real(_z)

    # Define prefix
    save_prefix = 'MYULA_wavelets_reg_param_{:.1e}_nsamples_{:.1e}_thinning_{:.1e}_frac_burn_{:.1e}'.format(
        reg_param, n_samples, thinning, frac_burnin
    )


    # Sampling alg params
    burnin = np.int64(n_samples * thinning * frac_burnin)
    X = x_init.clone()
    MC_X = np.zeros((n_samples, X.shape[1], X.shape[2]))
    logpi_thinning_trace = np.zeros((n_samples, 1))
    thinned_trace_counter = 0
    # thinning_step = np.int64(maxit/n_samples)

    nrmse_values = []
    psnr_values = []
    ssim_values = []
    logpi_eval = []

    # %%
    start_time = time.time()
    for i_x in tqdm(range(maxit)):

        # Update X
        X = luq.sampling.MYULA_kernel(
            X, delta, lmbd, grad_likelihood, prox_prior_cai, real_prox
        )

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
            logpi_eval.append(logPi(X).item())

            # collect thinned trace
            if np.mod(i_x - burnin, thinning) == 0:
                MC_X[thinned_trace_counter] = X.detach().cpu().numpy()
                logpi_thinning_trace[thinned_trace_counter] = logPi(X).item()
                thinned_trace_counter += 1

    end_time = time.time()
    elapsed = end_time - start_time    

    current_mean = post_meanvar.get_mean()
    current_var = post_meanvar.get_var().detach().cpu().squeeze()


    # %%
    # Compute the UQ plots
    superpix_sizes = [32,16,8,4,1]

    cmap = 'cubehelix'

    quantiles, st_dev_down, means_list = luq.map_uncertainty.compute_UQ(
        MC_X, superpix_sizes, alpha_prob
    )

    for it_3, pix_size in enumerate(superpix_sizes):

        # Plot UQ
        fig = plt.figure(figsize=(20,5))

        plt.subplot(131)
        ax = plt.gca()
        ax.set_title(f'Mean value, <Mean val>={np.mean(means_list[it_3]):.2e} pix size={pix_size:d}')
        im = ax.imshow(means_list[it_3], cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.set_yticks([]);ax.set_xticks([])

        plt.subplot(132)
        ax = plt.gca()
        ax.set_title(f'St Dev, <St Dev>={np.mean(st_dev_down[it_3]):.2e} pix size={pix_size:d}')
        im = ax.imshow(st_dev_down[it_3], cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.set_yticks([]);ax.set_xticks([])

        plt.subplot(133)
        LCI = quantiles[it_3][1,:,:] - quantiles[it_3][0,:,:]
        ax = plt.gca()
        ax.set_title(f'LCI, <LCI>={np.mean(LCI):.2e} pix size={pix_size:d}')
        im = ax.imshow(LCI, cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.set_yticks([]);ax.set_xticks([])
        plt.savefig(savefig_dir+save_prefix+'_UQ_pixel_size_{:d}.pdf'.format(pix_size))
        plt.close()



    # %%

    params = {
        'maxit': maxit,
        'n_samples': n_samples,
        'thinning': thinning,
        'frac_burnin': frac_burnin,
        # 'gamma': gamma,
        'delta': delta,
        'frac_delta': frac_delta,
        'reg_param': reg_param,
        # 'lambd_frac': lambd_frac,
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
    ax.set_title(f"Image MMSE (Regularization Cost {fun_prior(current_mean):.1f}, PSNR: {peak_signal_noise_ratio(to_tensor(MC_X_mean, device=device), torch_img).item():.2f})")
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

