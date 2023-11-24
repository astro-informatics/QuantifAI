import numpy as np
import skimage as ski
import torch
from functools import partial
from quantifai.utils import to_numpy, to_tensor, eval_snr
from quantifai.operators import DictionaryWv_torch, Operation2WaveletCoeffs_torch


def compute_UQ(MC_X_array, superpix_sizes=[32, 16, 8, 4, 1], alpha=0.05):
    """Compute uncertainty quantification stats.

    Args:

        MC_X_array (np.ndarray): Array with generated samples. Shape=(n_samples, nx, ny)
        superpix_sizes (list[int]): Superpixel sizes.
        alpha (float): Probability to compute the quantiles at `alpha/2` and `1-alpha/2`.

    Returns:

        quantiles (list(np.ndarray)): List corresponding to the superpix sizes.
            For each element we have the two computed quantiles, `alpha/2` and `1-alpha/2`.
        st_dev_down (list(np.ndarray)): List corresponding to the superpix sizes.
            For each element we have the standard deviation of each superpixel along the samples.
        means_list (list(np.ndarray)): List corresponding to the superpix sizes.
            For each element we have the mean of each superpixel along the samples.
    """

    n_samples = MC_X_array.shape[0]
    nx = MC_X_array.shape[1]
    ny = MC_X_array.shape[2]

    quantiles = []
    means_list = []
    st_dev_down = []

    p = np.array([alpha / 2, 1 - alpha / 2])

    for k, block_size in enumerate(superpix_sizes):
        downsample_array = np.zeros(
            [
                n_samples,
                np.int64(np.floor(nx / block_size)),
                np.int64(np.floor(ny / block_size)),
            ]
        )

        for j in range(n_samples):
            block_image = ski.measure.block_reduce(
                MC_X_array[j], block_size=(block_size, block_size), func=np.mean
            )
            if nx % block_size == 0:
                downsample_array[j] = block_image
            else:
                downsample_array[j] = block_image[:-1, :-1]

        # Compute quantiles for LCI
        quantiles.append(np.quantile(downsample_array, p, axis=0))
        # Compute pixel SD
        meanSample_down = np.mean(downsample_array, 0)
        second_moment_down = np.mean(downsample_array**2, 0)
        st_dev_down.append(np.sqrt(second_moment_down - meanSample_down**2))
        # Save the mean images
        means_list.append(meanSample_down)

    return quantiles, st_dev_down, means_list


class FastPixelUQ(object):
    """Class to compute the fast pixel UQ

    Args:
        x_map (torch.tensor): MAP estimation with shape (1, 1, xdim, ydim)
        myType (torch.dtype): torch type used by the torch-based potential function handle
        potential_handle (Callable): Potential function handle. The potential should be 
            the sum of the likelihood and prior terms. It can be set later.
        UQ_options (dict): Parameter dictionary. If it is None, the default parameters will be used.
    
    """
    def __init__(
        self,
        x_map,
        myType=torch.float32,
        potential_handle=None,
        UQ_options=None,
    ):
        self.x_map = x_map
        self.potential_handle = potential_handle
        self.myType = myType
        # HPD region parameters
        self.N = None
        self.gamma_alpha = None
        self.tau_alpha = None
        # Thresholded results
        self.selected_thresh = None
        self.thresh_img = None

        if UQ_options is None:
            print('Using default options for fast pixel UQ')
            self.UQ_options ={
                'alpha_prob': 0.01,         # Alpha for the HPD region
                'wavs_list': ['db8'],       # Wavelet type for UQ maps
                'levels': 4,                # Wavelet levels for UQ maps
                'start_interval': [0, 10],  # Parameter for UQ map bisection
                'iters': 5e2,               # Parameter for UQ map bisection
                'tol': 1e-2,                # Parameter for UQ map bisection
                'thresh_type': 'hard',      # Threshold type for fast UQ, options are 'hard' or 'soft'
            }
        else:
            self.UQ_options = UQ_options

        if potential_handle is not None:
            self.potential_handle_np = lambda _x : self.potential_handle(
                to_tensor(_x, dtype=myType)
            ).item()

        # Initialise the wavelet dictionaries
        # Define the l1 norm with dict psi
        self.Psi = DictionaryWv_torch(
            self.UQ_options['wavs_list'], self.UQ_options['levels']
        )
        self.oper2wavelet = Operation2WaveletCoeffs_torch(Psi=self.Psi)


    def set_quantifai_potential_fun_handle(self, lmbd, mu, CRR_model, likelihood):
        """Set the QuantifAI potential handle.

        Args:
            lmbd (float): QuantifAI's CRR-NN lambda parameter (regularisation strength).
            mu (float): QuantifAI's CRR-NN mu parameter.
            CRR_model (ConvexRidgeRegularizer torch.nn.Module): CRR-NN model
            likelihood (Grad Class): Unconstrained data-fidelity class

        """
        # Function handle for the potential
        def _potential_handle(_x, CRR_model, mu, lmbd):
            return (lmbd / mu) * CRR_model.cost(mu * _x) + likelihood.fun(_x)

        # Evaluation of the potential
        self.potential_handle = partial(
            _potential_handle, CRR_model=CRR_model, mu=mu, lmbd=lmbd
        )
        # Evaluation of the potential in numpy
        self.potential_handle_np = lambda _x : self.potential_handle(
            to_tensor(_x, dtype=self.myType)
        ).item()

    def compute_HPD_region(self):
        """Compute the high posterior density (HPD) region threshold approximation
        """
        if self.potential_handle is not None:
            self.N = to_numpy(self.x_map).size
            self.tau_alpha = np.sqrt(16*np.log(3/self.UQ_options['alpha_prob']))
            self.gamma_alpha = self.potential_handle(self.x_map).item() + self.tau_alpha*np.sqrt(self.N) + self.N
        else:
            raise ValueError('The potential_handle has not been defined. Set it manually in the initialisation or call self.set_quantifai_potential_fun_handle(...).')

    def compute_HPD_region_threshold(self):
        """ Compute wavelet threshold saturating the HPD region
        """
        # Clone MAP estimation and cast type for wavelet operations
        torch_map = torch.clone(self.x_map).to(torch.float64)

        def _potential_to_bisect(thresh, fun_np, oper2wavelet, torch_map):
            thresh_img = oper2wavelet.full_op_threshold_img(
                torch_map, thresh, thresh_type=self.UQ_options['thresh_type']
            )
            return self.gamma_alpha - fun_np(thresh_img)

        # Evaluation of the potential
        potential_to_bisect = partial(
            _potential_to_bisect,
            fun_np=self.potential_handle_np,
            oper2wavelet=self.oper2wavelet,
            torch_map=torch_map
        )

        selected_thresh, bisec_iters = bisection_method(
            potential_to_bisect,
            self.UQ_options['start_interval'],
            self.UQ_options['iters'],
            self.UQ_options['tol'],
            return_iters=True
        )
        thresh_img = self.oper2wavelet.full_op_threshold_img(
            torch_map, selected_thresh
        )

        self.selected_thresh = selected_thresh
        self.thresh_img = thresh_img

        return thresh_img, selected_thresh


    def compute_thresh_imgs_per_scale(self, x_gt=None, verbose=False):
        """Compute thresholded images as a function of scale.

        If a ground truth image is passed 

        Args:
            x_gt (np.ndarray): Ground truth image.
            verbose (bool): Verbosity.

        Returns:
            modif_img_list (list of np.ndarray): Thresholded MAP image corresponding to
                the computed HDP region as a function of scale. 
            GT_modif_img_list (list of np.ndarray): Ground truth image where the wavelet 
                coefficients of a single scale have been replaced by the MAP wavelet coefficients.
                Only if a ground truth image is used.
            SNR_at_lvl_list (list of float): Signal-to-noise-level of the thresholded image vs the 
                MAP reconstruction as a function of scale. Only if a ground truth image is used.
            SNR_at_lvl_map_vs_GT_list (list of float): Signal-to-noise-level of the MAP reconstruction 
                vs the ground truth as a function of scale. Only if a ground truth image is used.

        """
        # Clone MAP estimation and cast type for wavelet operations
        torch_map = torch.clone(self.x_map).to(torch.float64)
        if x_gt is not None:
            torch_x = to_tensor(x_gt).to(torch.float64)

        modif_img_list = []
        GT_modif_img_list = []
        SNR_at_lvl_list = []
        SNR_at_lvl_map_vs_GT_list = []

        for modif_level in range(self.UQ_options['levels']+1):

            op = lambda x1, x2: x2

            modif_img = self.oper2wavelet.full_op_two_img(
                torch.clone(torch_map),
                torch.clone(self.thresh_img),
                op,
                level=modif_level
            )
            modif_img_list.append(to_numpy(modif_img))

            if x_gt is not None:
                GT_modif_img = self.oper2wavelet.full_op_two_img(
                    torch.clone(torch_x),
                    torch.clone(torch_map),
                    op,
                    level=modif_level
                )
                if verbose:
                    print('SNR (thresh vs MAP) at lvl {:d}: {:f}'.format(
                        modif_level, eval_snr(to_numpy(torch_map), to_numpy(modif_img)))
                    )
                    print('SNR (MAP vs GT) at lvl {:d}: {:f}'.format(
                        modif_level, eval_snr(x_gt, to_numpy(GT_modif_img)))
                    )
                
                GT_modif_img_list.append(to_numpy(GT_modif_img))
                SNR_at_lvl_list.append(
                    eval_snr(x_gt, to_numpy(modif_img))
                )
                SNR_at_lvl_map_vs_GT_list.append(
                    eval_snr(x_gt, to_numpy(GT_modif_img))
                )
        
        if x_gt is not None:
            return modif_img_list, GT_modif_img_list, SNR_at_lvl_list, SNR_at_lvl_map_vs_GT_list
        else:
            return modif_img_list


    def run(self, x_gt, verbose=False):
        """Run the fast pixel UQ
        
        1st Compute HPD region.
        2nd Compute the threshold saturating the HPD region.
        3rd Compute the thresholded images per scale.
        
        
        Args:
            x_gt (np.ndarray): Ground truth image.
            verbose (bool): Verbosity.

        Returns:
            modif_img_list (list of np.ndarray): Thresholded MAP image corresponding to
                the computed HDP region as a function of scale. 
            GT_modif_img_list (list of np.ndarray): Ground truth image where the wavelet 
                coefficients of a single scale have been replaced by the MAP wavelet coefficients.
                Only if a ground truth image is used.
            SNR_at_lvl_list (list of float): Signal-to-noise-level of the thresholded image vs the 
                MAP reconstruction as a function of scale. Only if a ground truth image is used.
            SNR_at_lvl_map_vs_GT_list (list of float): Signal-to-noise-level of the MAP reconstruction 
                vs the ground truth as a function of scale. Only if a ground truth image is used.

        """
        self.compute_HPD_region()
        _, _ = self.compute_HPD_region_threshold()
        return self.compute_thresh_imgs_per_scale(x_gt, verbose)



def bisection_method(function, start_interval, iters, tol, return_iters=False):
    """Bisection method for locating minima of an abstract function

    Args:

        function (function): Loss function to bisect
        start_interval (list[int]): Initial lower and upper bounds
        iters (int): Maximum number of bisection iterations
        tol (double): Convergence tolerance of iterations
        return_iters (bool): return total number of iterations if True.

    Returns:

        Argument at which loss function is bisected
    """

    eta1 = start_interval[0]
    eta2 = start_interval[1]
    obj3 = function(eta2)
    if np.allclose(eta1, eta2, 1e-12):
        if return_iters:
            return eta1, 0
        else:
            return eta1
    if np.sign(function(eta1)) == np.sign(function(eta2)):
        print("[Bisection Method] There is no root in this range.")
        val = np.argmin(np.abs([eta1, eta2]))
        if return_iters:
            return [eta1, eta2][val], 2
        else:
            return [eta1, eta2][val]

    iters_cumul = 0
    for i in range(int(iters)):
        obj1 = function(eta1)
        eta3 = (eta2 + eta1) * 0.5
        obj3 = function(eta3)
        iters_cumul += 2
        if np.abs(eta1 - eta3) / np.abs(eta3) < tol:
            # if np.abs(obj3) < tol:
            if return_iters:
                return eta3, iters_cumul
            else:
                return eta3
        if np.sign(obj1) == np.sign(obj3):
            eta1 = eta3
        else:
            eta2 = eta3
    print("Did not converge... ", obj3)
    if return_iters:
        return eta3, iters_cumul
    else:
        return eta3


def create_local_credible_interval(
    x_sol,
    region_size,
    function,
    bound,
    iters,
    tol,
    bottom,
    top,
    verbose=0.0,
    return_iters=False,
):
    """Bisection method for finding credible intervals

    Args:

        x_sol (np.ndarray): Maximum a posteriori solution
        region_size (int): Super-pixel dimension
        function (function): Loss function to bisect
        bound (double): Level-set threshold at alpha % confidence
        iters (int): Maximum number of bisection iterations
        tol (double): Convergence tolerance of iterations
        bottom (double): lower bound on credible interval (>0)
        top (double): upper bound on credible interval (<0)
        return_iters (bool): return total number of iterations if True.

    Returns:

        Upper limit, lower limit, super-pixel mean, (iters_cumul)
    """

    region = np.zeros(x_sol.shape)
    print("Calculating credible interval for superpxiel: ", region.shape)
    if len(x_sol.shape) > 1:
        region[:region_size, :region_size] = 1.0
        dsizey, dsizex = int(x_sol.shape[0] / region_size), int(
            x_sol.shape[1] / region_size
        )
        error_p = np.zeros((dsizey, dsizex))
        error_m = np.zeros((dsizey, dsizex))
        mean = np.zeros((dsizey, dsizex))
        iters_cumul = 0
        for i in range(dsizey):
            for j in range(dsizex):
                mask = np.roll(
                    np.roll(region, shift=i * region_size, axis=0),
                    shift=j * region_size,
                    axis=1,
                )
                # x_sum = np.sum(np.ravel(x_sol[(mask.astype(bool))]))
                x_sum = np.mean(np.ravel(x_sol[(mask.astype(bool))]))
                mean[i, j] = x_sum

                def obj(eta):
                    return -bound + function(
                        x_sol * (1.0 - mask) + (x_sum + eta) * mask
                    )

                if return_iters:
                    error_p[i, j], bisec_iters = bisection_method(
                        obj, [0, top], iters, tol, return_iters
                    )
                    error_p[i, j] += x_sum
                    iters_cumul += bisec_iters
                else:
                    error_p[i, j] = (
                        bisection_method(obj, [0, top], iters, tol, return_iters)
                        + x_sum
                    )

                def obj(eta):
                    return -bound + function(
                        x_sol * (1.0 - mask) + (x_sum - eta) * mask
                    )

                if return_iters:
                    error_m[i, j], bisec_iters = bisection_method(
                        obj, [0, -bottom], iters, tol, return_iters
                    )
                    error_m[i, j] = error_m[i, j] * -1.0 + x_sum
                    iters_cumul += bisec_iters
                else:
                    error_m[i, j] = (
                        -bisection_method(obj, [0, -bottom], iters, tol, return_iters)
                        + x_sum
                    )

                if verbose > 0.0:
                    print(
                        "[Credible Interval] (%s, %s) has interval (%s, %s) with sum %s"
                        % (
                            i,
                            j,
                            error_m[i, j],
                            error_p[i, j],
                            x_sum,
                        )
                    )
    else:
        region[:region_size] = 1.0
        dsizey = int(x_sol.shape[0] / region_size)
        error_p = np.zeros((dsizey))
        error_m = np.zeros((dsizey))
        mean = np.zeros((dsizey))
        iters_cumul = 0

        for i in range(dsizey):
            mask = np.roll(region, shift=i * region_size, axis=0)
            # x_sum = np.sum(np.ravel(x_sol[(mask.astype(bool))]))
            x_sum = np.mean(np.ravel(x_sol[(mask.astype(bool))]))
            mean[i] = x_sum

            def obj(eta):
                return function(x_sol * (1.0 - mask) + eta * mask) - bound

            if return_iters:
                error_p[i], bisec_iters = bisection_method(
                    obj, [0, top], iters, tol, return_iters
                )
                iters_cumul += bisec_iters
            else:
                error_p[i] = bisection_method(obj, [0, top], iters, tol, return_iters)

            def obj(eta):
                return function(x_sol * (1.0 - mask) - eta * mask) - bound

            if return_iters:
                error_m[i], bisec_iters = bisection_method(
                    obj, [0, -bottom], iters, tol, return_iters
                )
                error_m[i] *= -1.0
                iters_cumul += bisec_iters
            else:
                error_m[i] = -bisection_method(
                    obj, [0, -bottom], iters, tol, return_iters
                )

            if verbose > 0.0:
                print(
                    "[Credible Interval] %s has interval (%s, %s) with sum %s"
                    % (
                        i,
                        error_m[i],
                        error_p[i],
                        x_sum,
                    )
                )

    if return_iters:
        return error_p, error_m, mean, iters_cumul
    else:
        return error_p, error_m, mean


def create_local_credible_interval_fast(
    x_sol, phi, psi, region_size, function, bound, iters, tol, bottom, top
):
    """Bisection method for finding credible intervals exploiting linearity

    Args:

        x_sol (np.ndarray): Maximum a posteriori solution
        phi (Linear operator): Sensing operator
        psi (Linear operator): Regularising operator (typically wavelets)
        region_size (int): Super-pixel dimension
        function (function): Loss function to bisect
        bound (double): Level-set threshold at alpha % confidence
        iters (int): Maximum number of bisection iterations
        tol (double): Convergence tolerance of iterations
        bottom (double): lower bound on credible interval
        top (double): upper bound on credible interval

    Returns:

        Upper limit, lower limit, super-pixel mean
    """

    region = np.zeros(x_sol.shape)
    print("Calculating credible interval for superpxiels: ", region.shape)
    if len(x_sol.shape) > 1:
        dsizey, dsizex = int(x_sol.shape[0] / region_size), int(
            x_sol.shape[1] / region_size
        )
        error_p = np.zeros((dsizey, dsizex))
        error_m = np.zeros((dsizey, dsizex))
        mean = np.zeros((dsizey, dsizex))
        mask = np.zeros(x_sol.shape)
        for i in range(dsizey):
            for j in range(dsizex):
                mask = mask * 0
                mask[
                    (i * region_size) : ((i + 1) * region_size),
                    (j * region_size) : ((j + 1) * region_size),
                ] = 1
                x_sum = np.mean(x_sol[mask > 0])
                mean[i, j] = x_sum
                buff = x_sol * (1.0 - mask) + mask * x_sum
                wav_sol = psi.dir_op(buff)
                data_sol = phi.dir_op(buff)
                wav_mask = psi.dir_op(mask)
                data_mask = phi.dir_op(mask)

                def obj(eta):
                    return (
                        function(data_sol, eta * data_mask, wav_sol, wav_mask * eta)
                        - bound
                    )

                error_p[i, j] = bisection_method(obj, [0, top], iters, tol)

                def obj(eta):
                    return (
                        function(data_sol, -eta * data_mask, wav_sol, wav_mask * -eta)
                        - bound
                    )

                error_m[i, j] = -bisection_method(obj, [0, -bottom], iters, tol)
                print(
                    "[Credible Interval] (%s, %s) has interval (%s, %s) with sum %s",
                    i,
                    j,
                    error_m[i, j],
                    error_p[i, j],
                    x_sum,
                )
    else:
        region[:region_size] = 1.0
        dsizey = int(x_sol.shape[0] / region_size)
        error_p = np.zeros((dsizey))
        error_m = np.zeros((dsizey))
        mean = np.zeros((dsizey))
        for i in range(dsizey):
            mask = np.roll(region, shift=i * region_size, axis=0)
            x_sum = np.sum(np.ravel(x_sol[(mask.astype(bool))]))
            mean[i] = x_sum

            def obj(eta):
                return function(x_sol * (1.0 - mask) + eta * mask) - bound

            error_p[i] = bisection_method(obj, [0, top], iters, tol)

            def obj(eta):
                return function(x_sol * (1.0 - mask) - eta * mask) - bound

            error_m[i] = -bisection_method(obj, [0, -bottom], iters, tol)
            print(
                "[Credible Interval] %s has interval (%s, %s) with sum %s",
                i,
                error_m[i],
                error_p[i],
                x_sum,
            )
    return error_p, error_m, mean
