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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(torch.cuda.current_device()))


from skimage.metrics import peak_signal_noise_ratio as psnr

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as tick

import scipy.io as sio
from astropy.io import fits
import skimage as ski

import quantifai as qai
from quantifai.utils import to_numpy, to_tensor
from convex_reg import utils as utils_cvx_reg

# %%

# Plot params
cmap = "cubehelix"
cbar_font_size = 18
box_font_size = 18

# Optimisation options for the MAP estimation
options = {"tol": 1e-5, "iter": 15000, "update_iter": 100, "record_iters": False}
# Save param
repo_dir = "./../../.."
base_savedir = "/disk/xray99/tl3/proj-convex-UQ/outputs/new_UQ_results/ungridded_results/CRR"
save_var_dir = base_savedir + "/vars/"
save_dir = base_savedir + "/figs/"

ungridded_vis_dir = repo_dir + "/data/meerkat_ungridded_vis/"

ungridded_vis_path_arr = [
    ungridded_vis_dir + "meerkat_simulation_1h_uv_only.npy",
    ungridded_vis_dir + "meerkat_simulation_2h_uv_only.npy",
    ungridded_vis_dir + "meerkat_simulation_4h_uv_only.npy",
    ungridded_vis_dir + "meerkat_simulation_8h_uv_only.npy",
]
    
ungridded_vis_times_arr = ["1h", "2h","4h","8h"]

img_name_arr = ["CYN", "M31", "3c288", "W28"]
vmin_log_arr = [-3.0,-2.0,-2.0,-2.0,]

computing_time_arr_uq = []


# CRR load parameters
sigma_training = 5
t_model = 5
CRR_dir_name = repo_dir + "/trained_models/"
# CRR parameters
reg_param_list = [
    1e4,
    1.4e4,
    1.9e4,
    2.25e4,
]
mu = 20

# Define my torch types (CRR requires torch.float32)
myType = torch.float32
myComplexType = torch.complex64

# Parameters
alpha_prob = 0.01

# Define the wavelet parameters for UQ maps
wavs_list = ["db8"]
levels = 4
# Parameters for UQ map
start_interval = [0, 10]
iters = 5e2
tol = 1e-2

model_prefix = "-CRR"
input_snr = 30.0


save_fig_vals = True



for it in range(len(img_name_arr)):

    for it_vis in range(len(ungridded_vis_times_arr)):
        
        reg_param = reg_param_list[it_vis]
        ungridded_vis_time = ungridded_vis_times_arr[it_vis]
        ungridded_vis_path = ungridded_vis_path_arr[it_vis]
            
        # Set paths
        if model_prefix == "-CRR":
            img_name = img_name_arr[it]
            vmin_log = vmin_log_arr[it]
        
        computing_time_arr_uq = []
        n_iters_uq = []
            
        n_iter_optim = []
        computing_time_optim = []

        map_snr_arr = [] 

        # Load image and mask
        img, mat_mask = qai.helpers.load_imgs(img_name, repo_dir)
        # Aliases
        x = img
        ground_truth = img

        # Convert Torch
        torch_img = torch.tensor(np.copy(img), dtype=myType, device=device).reshape(
            (1, 1) + img.shape
        )

        # Load uv data
        uv_data = np.load(ungridded_vis_path, allow_pickle=True)[()]
        uv = np.concatenate((uv_data['uu'].reshape(-1,1), uv_data['vv'].reshape(-1,1)), axis=1)
        torch_uv = torch.tensor(uv.T, device=device, dtype=myType)

        # Init NUFFT op
        phi = qai.operators.KbNuFFT2d_torch(
            uv=torch_uv,
            im_shape=img.shape,
            device=device,
            interp_points=6,
            k_oversampling=2,
            myType=myType,
            myComplexType=myComplexType
        )
            
        
        y = phi.dir_op(torch_img).detach().cpu().squeeze().numpy()

        # Define X Cai noise level
        eff_sigma = qai.helpers.compute_complex_sigma_noise(y, input_snr)
        sigma = eff_sigma * np.sqrt(2)

        # Generate noise
        rng = np.random.default_rng(seed=0)
        n_re = rng.normal(0, eff_sigma, y[y != 0].shape)
        n_im = rng.normal(0, eff_sigma, y[y != 0].shape)
        # Add noise
        y[y != 0] += n_re + 1.0j * n_im

        # Observation
        torch_y = torch.tensor(np.copy(y), device=device, dtype=myComplexType)[None, None, :]

        x_init = torch.abs(phi.adj_op(torch_y))
        # Define the likelihood
        likelihood = qai.operators.L2Norm_torch(
            sigma=sigma,
            data=torch_y,
            Phi=phi,
            im_shape=x_init.shape
        )
        # Lipschitz constant computed automatically by likelihood, stored in likelihood.beta
        # Define real prox
        prox_op = qai.operators.RealProx_torch()

        # Load CRR model
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)

        exp_name = f"Sigma_{sigma_training}_t_{t_model}/"
        CRR_model = utils_cvx_reg.load_model(
            CRR_dir_name + exp_name, "cuda:0", device_type="gpu"
        )

        print(f"Numbers of parameters before prunning: {CRR_model.num_params}")
        CRR_model.prune()
        print(f"Numbers of parameters after prunning: {CRR_model.num_params}")

        # L_CRR = CRR_model.L.detach().cpu().squeeze().numpy()
        # print(f"Lipschitz bound {L_CRR:.3f}")

        # [not required] intialize the eigen vector of dimension (size, size) associated to the largest eigen value
        CRR_model.initializeEigen(size=100)
        # compute bound via a power iteration which couples the activations and the convolutions
        CRR_model.precise_lipschitz_bound(n_iter=100)
        # the bound is stored in the model
        L_CRR = CRR_model.L.data.item()
        print(f"Lipschitz bound {L_CRR:.3f}")

        ## Compute MAP solution
        # Prior parameters
        lmbd = reg_param

        # Compute stepsize
        alpha = 0.98 / (likelihood.beta + mu * lmbd * L_CRR)

        wall_time_1 = time.time()

        x_hat, total_it = qai.optim.FISTA_CRR_torch(
            x_init=x_init,
            options=options,
            likelihood=likelihood,
            prox_op=prox_op,
            CRR_model=CRR_model,
            alpha=alpha,
            lmbd=lmbd,
            mu=mu,
            return_iter_num=True,
        )
            
        wall_time_2 = time.time()

        computing_time_optim.append(wall_time_2 - wall_time_1)
        n_iter_optim.append(total_it)
        
        # Save MAP
        np_x_hat = to_numpy(x_hat)
        np_x = np.copy(x)
        # Evaluate performance
        print(img_name, " PSNR: ", psnr(np_x, np_x_hat, data_range=np_x.max() - np_x.min()))
        print(img_name, " SNR: ", qai.utils.eval_snr(x, np_x_hat))
            
        map_snr_arr.append(qai.utils.eval_snr(x, np_x_hat))

        # Function handle for the potential
        def _fun(_x, CRR_model, mu, lmbd):
            return (lmbd / mu) * CRR_model.cost(mu * _x) + likelihood.fun(_x)

        # Evaluation of the potential
        fun = partial(_fun, CRR_model=CRR_model, mu=mu, lmbd=lmbd)
        # Evaluation of the potential in numpy
        fun_np = lambda _x: fun(qai.utils.to_tensor(_x, dtype=myType)).item()

        # Compute HPD region bound
        N = np_x_hat.size
        tau_alpha = np.sqrt(16 * np.log(3 / alpha_prob))
        gamma_alpha = fun(x_hat).item() + tau_alpha * np.sqrt(N) + N

        # Define the wavelet dict
        # Define the l1 norm with dict psi
        Psi = qai.operators.DictionaryWv_torch(wavs_list, levels, shape=torch_img.shape)
        oper2wavelet = qai.operators.Operation2WaveletCoeffs_torch(Psi=Psi)

        # Clone MAP estimation and cast type for wavelet operations
        torch_map = torch.clone(x_hat).to(torch.float64)
        torch_x = to_tensor(np_x).to(torch.float64)

        def _potential_to_bisect(thresh, fun_np, oper2wavelet, torch_map):
            thresh_img = oper2wavelet.full_op_threshold_img(
                torch_map, thresh, thresh_type="hard"
            )

            return gamma_alpha - fun_np(thresh_img)

        # Evaluation of the potential
        potential_to_bisect = partial(
            _potential_to_bisect,
            fun_np=fun_np,
            oper2wavelet=oper2wavelet,
            torch_map=torch_map,
        )

        wall_time_1 = time.time()

        selected_thresh, bisec_iters = qai.map_uncertainty.bisection_method(
            potential_to_bisect, start_interval, iters, tol, return_iters=True
        )
        select_thresh_img = oper2wavelet.full_op_threshold_img(torch_map, selected_thresh)
        wall_time_2 = time.time()
        # Save iteration number for pixel UQ
        n_iters_uq.append(bisec_iters)
        print("Pixel UQ required ", bisec_iters, " iterations to converge.")

        # Save time
        computing_time_arr_uq.append(wall_time_2 - wall_time_1)

        print("selected_thresh: ", selected_thresh)
        print("gamma_alpha: ", gamma_alpha)
        print("MAP image: ", fun_np(torch_map.squeeze()))
        print("thresholded image: ", fun_np(select_thresh_img))

        # Compute SNR
        x_dirty = to_numpy(torch.abs(phi.adj_op(torch_y))) # This is x_init
        dirty_snr = qai.utils.eval_snr(np_x, x_dirty)
        # Plot dirty reconstruction
        fig = plt.figure(figsize=(5, 5), dpi=200)
        axs = plt.gca()
        plt_im = axs.imshow(np.log10(abs(x_dirty)), cmap=cmap, vmin=vmin_log, vmax=0)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(plt_im, cax=cax)
        cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter("%.1f"))
        cbar.ax.tick_params(labelsize=cbar_font_size)
        axs.set_yticks([])
        axs.set_xticks([])
        plt.tight_layout()
        if save_fig_vals:
            plt.savefig(
                "{:s}{:s}{:s}{:s}{:s}".format(
                    save_dir, img_name, model_prefix, ungridded_vis_time, "-newPixelUQ-dirty.pdf"
                ),
                bbox_inches="tight",
                dpi=200,
            )
        plt.close()

        # Compute SNR
        x_dirty = to_numpy(torch.abs(phi.adj_op(torch_y))) # This is x_init
        dirty_snr = qai.utils.eval_snr(np_x, x_dirty)
        # Plot dirty reconstruction with SNR box
        fig = plt.figure(figsize=(5, 5), dpi=200)
        axs = plt.gca()
        plt_im = axs.imshow(np.log10(abs(x_dirty)), cmap=cmap, vmin=vmin_log, vmax=0)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(plt_im, cax=cax)
        cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter("%.1f"))
        cbar.ax.tick_params(labelsize=cbar_font_size)
        axs.set_yticks([])
        axs.set_xticks([])
        textstr = r"$\mathrm{SNR}=%.2f$ dB" % (np.mean(dirty_snr))
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        axs.text(
            0.05,
            0.95,
            textstr,
            transform=axs.transAxes,
            fontsize=box_font_size,
            verticalalignment="top",
            bbox=props,
        )
        plt.tight_layout()
        if save_fig_vals:
            plt.savefig(
                "{:s}{:s}{:s}{:s}{:s}".format(
                    save_dir, img_name, model_prefix, ungridded_vis_time, "-newPixelUQ-dirty_SNRbox.pdf"
                ),
                bbox_inches="tight",
                dpi=200,
            )
        plt.close()

        # Compute SNR
        map_snr = qai.utils.eval_snr(np_x, np_x_hat)
        # Plot MAP with SNR box
        fig = plt.figure(figsize=(5, 5), dpi=200)
        axs = plt.gca()
        im_log = np.log10(np.abs(np_x_hat))
        plt_im = axs.imshow(im_log, cmap=cmap, vmin=vmin_log, vmax=0)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(plt_im, cax=cax)
        cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter("%.2f"))
        cbar.ax.tick_params(labelsize=cbar_font_size)
        axs.set_yticks([])
        axs.set_xticks([])
        textstr = r"$\mathrm{SNR}=%.2f$ dB" % (np.mean(map_snr))
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        axs.text(
            0.05,
            0.95,
            textstr,
            transform=axs.transAxes,
            fontsize=box_font_size,
            verticalalignment="top",
            bbox=props,
        )
        plt.tight_layout()
        if save_fig_vals:
            plt.savefig(
                "{:s}{:s}{:s}{:s}{:s}".format(
                    save_dir, img_name, model_prefix, ungridded_vis_time, "-newPixelUQ-MAP_SNRbox.pdf"
                ),
                bbox_inches="tight",
                dpi=200,
            )
        plt.close()

        # Plot MAP
        fig = plt.figure(figsize=(5, 5), dpi=200)
        axs = plt.gca()
        im_log = np.log10(np.abs(np_x_hat))
        plt_im = axs.imshow(im_log, cmap=cmap, vmin=vmin_log, vmax=0)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(plt_im, cax=cax)
        cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter("%.2f"))
        cbar.ax.tick_params(labelsize=cbar_font_size)
        axs.set_yticks([])
        axs.set_xticks([])
        plt.tight_layout()
        if save_fig_vals:
            plt.savefig(
                "{:s}{:s}{:s}{:s}{:s}".format(
                    save_dir, img_name, model_prefix, ungridded_vis_time, "-newPixelUQ-MAP.pdf"
                ),
                bbox_inches="tight",
                dpi=200,
            )
        plt.close()

        # Plot Thresholded image
        fig = plt.figure(figsize=(5, 5), dpi=200)
        axs = plt.gca()
        im_log = np.log10(np.abs(to_numpy(select_thresh_img)))
        plt_im = axs.imshow(im_log, cmap=cmap, vmin=vmin_log, vmax=0)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(plt_im, cax=cax)
        cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter("%.2f"))
        cbar.ax.tick_params(labelsize=cbar_font_size)
        axs.set_yticks([])
        axs.set_xticks([])
        plt.tight_layout()
        if save_fig_vals:
            plt.savefig(
                "{:s}{:s}{:s}{:s}{:s}".format(
                    save_dir, img_name, model_prefix, ungridded_vis_time,"-newPixelUQ-ThresholdedImage.pdf"
                ),
                bbox_inches="tight",
                dpi=200,
            )
        plt.close()

        # Plot MAP - Thresholded error
        fig = plt.figure(figsize=(5, 5), dpi=200)
        axs = plt.gca()
        im_log = np.log10(np.abs(to_numpy(torch_map - select_thresh_img)))
        plt_im = axs.imshow(im_log, cmap=cmap, vmin=vmin_log - 2, vmax=0)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(plt_im, cax=cax)
        cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter("%.2f"))
        cbar.ax.tick_params(labelsize=cbar_font_size)
        axs.set_yticks([])
        axs.set_xticks([])
        plt.tight_layout()
        if save_fig_vals:
            plt.savefig(
                "{:s}{:s}{:s}{:s}{:s}".format(
                    save_dir,
                    img_name,
                    model_prefix,
                    ungridded_vis_time,
                    "-newPixelUQ-MAP_thresholded_error.pdf",
                ),
                bbox_inches="tight",
                dpi=200,
            )
        plt.close()

        # Plot Ground truth - MAP
        fig = plt.figure(figsize=(5, 5), dpi=200)
        axs = plt.gca()
        im_log = np.log10(np.abs(np_x - np_x_hat))
        plt_im = axs.imshow(im_log, cmap=cmap, vmin=vmin_log - 2, vmax=0)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(plt_im, cax=cax)
        cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter("%.2f"))
        cbar.ax.tick_params(labelsize=cbar_font_size)
        axs.set_yticks([])
        axs.set_xticks([])
        plt.tight_layout()
        if save_fig_vals:
            plt.savefig(
                "{:s}{:s}{:s}{:s}{:s}".format(
                    save_dir, img_name, model_prefix, ungridded_vis_time,"-newPixelUQ-GT_MAP_error.pdf"
                ),
                bbox_inches="tight",
                dpi=200,
            )
        plt.close()

        modif_img_list = []
        GT_modif_img_list = []
        SNR_at_lvl_list = []
        SNR_at_lvl_map_vs_GT_list = []

        for modif_level in range(levels + 1):
            op = lambda x1, x2: x2

            modif_img = oper2wavelet.full_op_two_img(
                torch.clone(torch_map),
                torch.clone(select_thresh_img),
                op,
                level=modif_level,
            )
            GT_modif_img = oper2wavelet.full_op_two_img(
                torch.clone(torch_x), torch.clone(torch_map), op, level=modif_level
            )
            print(
                "SNR (thresh vs MAP) at lvl {:d}: {:f}".format(
                    modif_level,
                    qai.utils.eval_snr(to_numpy(torch_map), to_numpy(modif_img)),
                )
            )
            print(
                "SNR (MAP vs GT) at lvl {:d}: {:f}".format(
                    modif_level,
                    qai.utils.eval_snr(to_numpy(torch_x), to_numpy(GT_modif_img)),
                )
            )
            modif_img_list.append(to_numpy(modif_img))
            GT_modif_img_list.append(to_numpy(GT_modif_img))
            SNR_at_lvl_list.append(
                qai.utils.eval_snr(to_numpy(torch_map), to_numpy(modif_img))
            )
            SNR_at_lvl_map_vs_GT_list.append(
                qai.utils.eval_snr(to_numpy(torch_x), to_numpy(GT_modif_img))
            )

            # Plot MAP - Thresholded error
            fig = plt.figure(figsize=(5, 5), dpi=200)
            axs = plt.gca()
            im_log = np.log10(np.abs(to_numpy(torch_map - modif_img)))
            plt_im = axs.imshow(im_log, cmap=cmap, vmin=vmin_log - 2, vmax=0)
            divider = make_axes_locatable(axs)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(plt_im, cax=cax)
            cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter("%.2f"))
            cbar.ax.tick_params(labelsize=cbar_font_size)
            axs.set_yticks([])
            axs.set_xticks([])
            plt.tight_layout()
            if save_fig_vals:
                plt.savefig(
                    "{:s}{:s}{:s}{:s}{:s}{:d}{:s}".format(
                        save_dir,
                        img_name,
                        model_prefix,
                        ungridded_vis_time,
                        "-newPixelUQ-MAP_thresholded_error_level_",
                        modif_level,
                        ".pdf",
                    ),
                    bbox_inches="tight",
                    dpi=200,
                )
            plt.close()

            # Plot GT - MAP error
            fig = plt.figure(figsize=(5, 5), dpi=200)
            axs = plt.gca()
            im_log = np.log10(np.abs(np_x - to_numpy(GT_modif_img)))
            plt_im = axs.imshow(im_log, cmap=cmap, vmin=vmin_log - 2, vmax=0)
            divider = make_axes_locatable(axs)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(plt_im, cax=cax)
            cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter("%.2f"))
            cbar.ax.tick_params(labelsize=cbar_font_size)
            axs.set_yticks([])
            axs.set_xticks([])
            plt.tight_layout()
            if save_fig_vals:
                plt.savefig(
                    "{:s}{:s}{:s}{:s}{:s}{:d}{:s}".format(
                        save_dir,
                        img_name,
                        model_prefix,
                        ungridded_vis_time,
                        "-newPixelUQ-GT_MAP_error_level_",
                        modif_level,
                        ".pdf",
                    ),
                    bbox_inches="tight",
                    dpi=200,
                )
            plt.close()

        config_dict = {
            "sigma_training": sigma_training,
            "t_model": t_model,
            "reg_param": reg_param,
            "mu": mu,
            "alpha_prob": alpha_prob,
            "wavs_list": wavs_list,
            "levels": levels,
            "start_interval": start_interval,
            "iters": iters,
            "tol": tol,
            "optim_options": options,
        }
        save_dict = {
            "gt": np_x,
            "map": np_x_hat,
            "x_dirty": x_dirty,
            "thresholded_img": to_numpy(select_thresh_img),
            "map_thresh_error_at_level": np.array(modif_img_list),
            "gt_map_error_at_level": np.array(GT_modif_img_list),
            "SNR_at_level": np.array(SNR_at_lvl_list),
            "SNR_at_lvl_map_vs_GT": np.array(SNR_at_lvl_map_vs_GT_list),
            "computing_time_arr_uq": computing_time_arr_uq,
            "n_iters_uq": n_iters_uq,
            "computing_time_optim": computing_time_optim,
            "n_iter_optim": n_iter_optim,
            "map_snr_arr": map_snr_arr, 
            "config_dict": config_dict,
        }

        # We will overwrite the dict with new results
        try:
            saving_var_path = "{:s}{:s}{:s}{:s}{:s}".format(
                save_var_dir,
                img_name,
                model_prefix,
                ungridded_vis_time,
                "-new_pixel_UQ_vars.npy",
            )
            if save_fig_vals:
                if os.path.isfile(saving_var_path):
                    os.remove(saving_var_path)
                np.save(saving_var_path, save_dict, allow_pickle=True)
        except Exception as e:
            print("Could not save vairables. Exception caught: ", e)



# %%


# %%
for it in range(len(img_name_arr)):
    for it_vis in range(len(ungridded_vis_times_arr)):

        ungridded_vis_time = ungridded_vis_times_arr[it_vis]
            

        # Set paths
        if model_prefix == "-CRR":
            img_name = img_name_arr[it]
            # save_dir = CRR_save_dir
            

        saving_var_path = "{:s}{:s}{:s}{:s}{:s}".format(
            save_var_dir,
            img_name,
            model_prefix,
            ungridded_vis_time,
            "-new_pixel_UQ_vars.npy",
        )

        data = np.load(saving_var_path, allow_pickle=True)[()]

        print("\n\n", img_name, "obs time: ", ungridded_vis_time)
            
        print("MAP SNR: ", data['map_snr_arr'])
        print("Optim iterations: ", data['n_iter_optim'])
        print("Optim computing time: ", data['computing_time_optim'])
            
        print("UQ iterations: ", data['n_iters_uq'])
        print("UQ computing time: ", data['computing_time_arr_uq'])
        
        print("***")

        print("SNR (MAP vs GT): \t\t\t", qai.utils.eval_snr(data["gt"], data["map"]))
        for modif_level in range(levels + 1):
            print(
                "SNR (MAP vs GT) at lvl {:d}: \t\t\t{:.2f}".format(
                    modif_level, data["SNR_at_lvl_map_vs_GT"][modif_level]
                )
            )
        print(
            "SNR (thresh vs MAP): \t\t\t",
            qai.utils.eval_snr(data["map"], data["thresholded_img"]),
        )
        for modif_level in range(levels + 1):
            print(
                "SNR (thresh vs MAP) at lvl {:d}: \t\t\t{:.2f}".format(
                    modif_level, data["SNR_at_level"][modif_level]
                )
            )



