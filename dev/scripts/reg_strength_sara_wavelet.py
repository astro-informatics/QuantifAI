# %%
import os
import numpy as np
import time as time

# Import torch and select GPU
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

# Plot functions
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Radio and convex reg functions
import quantifai as qai
from quantifai.utils import to_numpy, to_tensor
from convex_reg import utils as utils_cvx_reg

import skimage as ski
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# %%



# %% [markdown]
# ## Run

# %%


# Optimisation options for the MAP estimation
options = {"tol": 1e-5, "iter": 1500, "update_iter": 100, "record_iters": False}

# Save param
repo_dir = "./../.."

# Test image name from ['M31', 'W28', 'CYN', '3c288']
img_name_list = ['M31', 'W28', 'CYN', '3c288']
# img_name = "CYN"
# Input noise level
input_snr = 30.0

# Define my torch types (CRR requires torch.float32)
myType = torch.float64
myComplexType = torch.complex128

# CRR load parameters
sigma_training = 5
t_model = 5
CRR_dir_name = "./../../trained_models/"
# CRR parameters
lmbd = 5e4  # lambda parameter
mu = 20

alpha_prob = 0.01

wavs_list = ["db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8", "self"]
levels = 4
# reg_param = 1e2

# Compute the MAP-based UQ plots
superpix_MAP_sizes = [32, 16, 8, 4]
# Clipping values for MAP-based LCI. Set as None for no clipping
clip_high_val = 1.0
clip_low_val = 0.0

# Compute the sampling UQ plots
superpix_sizes = [32, 16, 8, 4, 1]

# LCI algorithm parameters (bisection)
LCI_iters = 200
LCI_tol = 1e-4
LCI_bottom = -10
LCI_top = 10



# %%


map_potential_list = []
gamma_alpha_list = []
likelihood_map_potential_list = []
prior_map_potential_list = []
SNR_list = []
PSNR_list = []


# %%
reg_param_list = np.logspace(np.log10(20), np.log10(5e4), num=30, endpoint=True, base=10.0)
# reg_param_list[0] = 7e1 
# reg_param_list[1] = 8e1 
# reg_param_list[2] = 9e1
# reg_param_list[3] = 1e2
# reg_param_list[4] = 1.1e2
# reg_param_list[25] = 1e4

reg_param_list

# %%


# %%
# Parameters

result_dict_list = []

for _img_name in img_name_list:

    img_name = _img_name

    result_dict = {}

    map_potential_list = []
    gamma_alpha_list = []
    likelihood_map_potential_list = []
    prior_map_potential_list = []
    SNR_list = []
    PSNR_list = []
    lci_mean_list = []

    for _reg_param in reg_param_list:

        reg_param = _reg_param

        # Load image and mask
        img, mat_mask = qai.helpers.load_imgs(img_name, repo_dir)

        # Aliases
        x = img
        ground_truth = img

        torch_img = torch.tensor(np.copy(img), dtype=myType, device=device).reshape(
            (1, 1) + img.shape
        )

        phi = qai.operators.MaskedFourier_torch(
            shape=img.shape, ratio=0.5, mask=mat_mask, norm="ortho", device=device
        )

        y = phi.dir_op(torch_img).detach().cpu().squeeze().numpy()

        # Define noise level
        eff_sigma = qai.helpers.compute_complex_sigma_noise(y, input_snr)
        sigma = eff_sigma * np.sqrt(2)

        # Generate noise
        rng = np.random.default_rng(seed=0)
        n_re = rng.normal(0, eff_sigma, y[y != 0].shape)
        n_im = rng.normal(0, eff_sigma, y[y != 0].shape)
        # Add noise
        y[y != 0] += n_re + 1.0j * n_im

        # Observation
        torch_y = torch.tensor(np.copy(y), device=device, dtype=myComplexType).reshape(
            (1,) + img.shape
        )
        # Generate first guess
        x_init = torch.abs(phi.adj_op(torch_y))



        # Define the likelihood
        likelihood = qai.operators.L2Norm_torch(
            sigma=sigma,
            data=torch_y,
            Phi=phi,
        )
        # Lipschitz constant computed automatically by likelihood, stored in likelihood.beta

        # Define real prox
        cvx_set_prox_op = qai.operators.RealProx_torch()

        # Define the wavelet dict
        # Define the l1 norm with dict psi
        psi = qai.operators.DictionaryWv_torch(wavs_list, levels, shape=torch_img.shape)
        reg_prox_op = qai.operators.L1Norm_torch(1.0, psi, op_to_coeffs=True)
        reg_prox_op.gamma = reg_param


        # Compute stepsize
        alpha = 0.98 / likelihood.beta


        # Run the optimisation
        x_hat, diagnostics = qai.optim.FISTA_torch(
            x_init,
            options=options,
            likelihood=likelihood,
            cvx_set_prox_op=cvx_set_prox_op,
            reg_prox_op=reg_prox_op,
            alpha=alpha,
            tau=alpha,
            viewer=None,
        )


        # Convert to numpy
        np_x_init = to_numpy(x_init)
        x_map = x_hat.clone()
        x_gt = np.copy(x)
        np_x_gt = np.copy(x)
        np_x_map = to_numpy(x_map)


        print('\nMAP reg_param: ', reg_param)
        print('Image: ', img_name)
        print('PSNR: {},\n SNR: {}, SSIM: {}'.format(
            round(psnr(x_gt, np_x_map, data_range=x_gt.max()-x_gt.min()), 2),
            round(qai.utils.eval_snr(x_gt, np_x_map), 2),
            round(ssim(x_gt, np_x_map, data_range=x_gt.max()-x_gt.min()), 2),
        ))

        SNR_list.append(qai.utils.eval_snr(x_gt, np_x_map))
        PSNR_list.append(psnr(x_gt, np_x_map, data_range=x_gt.max()-x_gt.min()))


        # To tensor
        x_map_torch = x_map.clone() # to_tensor(x_map)

        # Compute stepsize
        alpha = 0.98 / likelihood.beta

        #function handles for the hypothesis test

        # Evaluation of the potentials
        # Prior potential
        prior_fun = lambda _x : reg_prox_op._fun_coeffs(reg_prox_op.dir_op(_x))
        # Posterior potential
        fun = lambda _x : likelihood.fun(_x) +  prior_fun(_x)
        # Evaluation of the potential in numpy
        fun_np = lambda _x : fun(to_tensor(_x, dtype=myType)).item()

        # Compute HPD region bound
        N = np_x_map.size
        tau_alpha = np.sqrt(16*np.log(3/alpha_prob))
        gamma_alpha = fun(x_map_torch).item() + tau_alpha * np.sqrt(N) + N

        print('gamma_alpha: ', gamma_alpha)
        print('fun(x_map).item(): ', fun(x_map_torch).item())
        print('tau_alpha*np.sqrt(N) + N: ', tau_alpha*np.sqrt(N) + N)

        # Compute potential
        map_potential = fun(x_map_torch).item()

        # Decompose potentials
        map_likelihood_potential = likelihood.fun(x_map_torch).item()
        map_prior_potential = prior_fun(x_map_torch).item()

        # Print values
        print(img_name, '_gamma_alpha: ', gamma_alpha)
        print(img_name, '-MAP_potential: ', map_potential)

        # Save values
        map_potential_list.append(map_potential)
        gamma_alpha_list.append(gamma_alpha)

        # Save decomposed potentials
        likelihood_map_potential_list.append(map_likelihood_potential)
        prior_map_potential_list.append(map_prior_potential)


        ### MAP-based UQ

        # Define prior potential
        fun_prior = lambda _x: reg_prox_op._fun_coeffs(reg_prox_op.dir_op(_x))
        # Define posterior potential
        loss_fun_torch = lambda _x: likelihood.fun(_x) + fun_prior(_x)
        # Numpy version of the posterior potential
        loss_fun_np = (
            lambda _x: likelihood.fun(qai.utils.to_tensor(_x, dtype=myType)).item()
            + fun_prior(qai.utils.to_tensor(_x, dtype=myType)).item()
        )

        # Compute HPD region bound
        N = np_x_map.size
        tau_alpha = np.sqrt(16 * np.log(3 / alpha_prob))
        gamma_alpha = loss_fun_torch(x_hat).item() + tau_alpha * np.sqrt(N) + N

        # Compute the LCI
        error_p_arr = []
        error_m_arr = []
        mean_img_arr = []
        computing_time = []
        lci_mean = []

        x_init_np = qai.utils.to_numpy(x_init)

        # Compute ground truth block
        gt_mean_img_arr = []
        for superpix_size in superpix_MAP_sizes:
            mean_image = ski.measure.block_reduce(
                np.copy(img), block_size=(superpix_size, superpix_size), func=np.mean
            )
            gt_mean_img_arr.append(mean_image)

        # Define prefix
        save_MAP_prefix = "{:s}_wavelets_UQ_MAP_reg_param_{:.1e}".format(
            img_name, reg_param
        )

        for it_pixs, superpix_size in enumerate(superpix_MAP_sizes):
            pr_time_1 = time.process_time()
            wall_time_1 = time.time()

            error_p, error_m, mean = qai.map_uncertainty.create_local_credible_interval(
                x_sol=np_x_map,
                region_size=superpix_size,
                function=loss_fun_np,
                bound=gamma_alpha,
                iters=LCI_iters,
                tol=LCI_tol,
                bottom=LCI_bottom,
                top=LCI_top,
            )
            pr_time_2 = time.process_time()
            wall_time_2 = time.time()
            # Add values to array to save it later
            error_p_arr.append(np.copy(error_p))
            error_m_arr.append(np.copy(error_m))
            mean_img_arr.append(np.copy(mean))
            computing_time.append((pr_time_2 - pr_time_1, wall_time_2 - wall_time_1))
            # Clip plot values
            error_length = qai.utils.clip_matrix(
                np.copy(error_p), clip_low_val, clip_high_val
            ) - qai.utils.clip_matrix(np.copy(error_m), clip_low_val, clip_high_val)
            # Recover the ground truth mean
            gt_mean = gt_mean_img_arr[it_pixs]

            lci_mean.append(np.mean(error_length))
            print(img_name, '_lci_mean_', superpix_size,': ', np.mean(error_length))


        lci_mean_list.append(lci_mean)


    result_dict = {
        'map_potential_list': map_potential_list,
        'gamma_alpha_list': gamma_alpha_list,
        'likelihood_map_potential_list': likelihood_map_potential_list,
        'prior_map_potential_list': prior_map_potential_list,
        'SNR_list': SNR_list,
        'PSNR_list': PSNR_list,
        'lci_mean_list': lci_mean_list,
    }
    # Save dict in result list
    result_dict_list.append(result_dict)


save_path = "/disk/xray0/tl3/repos/QuantifAI/dev/tmp_results/reg_strength_sara/reg_strength_sara.npy"

np.save(save_path, result_dict_list, allow_pickle=True)    


# %%



# %%


# %%


# %%


# %%



# %%


# %%


# %%



