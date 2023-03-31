import numpy as np
import logging
import skimage as ski

# logger = logging.getLogger("Optimus Primal")


def compute_UQ(MC_X_array, superpix_sizes=[32,16,8,4,1], alpha=0.05):
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

    p = np.array([alpha/2, 1-alpha/2])

    for k, block_size  in enumerate(superpix_sizes):
        
        downsample_array= np.zeros([
            n_samples,
            np.int64(np.floor(nx/block_size)),
            np.int64(np.floor(ny/block_size))
        ])

        for j in range(n_samples):
            block_image = ski.measure.block_reduce(
                MC_X_array[j], block_size=(block_size, block_size), func=np.mean
            )
            if nx % block_size == 0:
                downsample_array[j] = block_image
            else:
                downsample_array[j] = block_image[:-1,:-1]

        # Compute quantiles for LCI
        quantiles.append(np.quantile(downsample_array, p, axis=0))
        # Compute pixel SD
        meanSample_down = np.mean(downsample_array, 0)
        second_moment_down = np.mean(downsample_array**2, 0)
        st_dev_down.append(np.sqrt(second_moment_down - meanSample_down**2))
        # Save the mean images
        means_list.append(meanSample_down)

    return quantiles, st_dev_down, means_list


def bisection_method(function, start_interval, iters, tol):
    """Bisection method for locating minima of an abstract function

    Args:

        function (function): Loss function to bisect
        start_interval (list[int]): Initial lower and upper bounds
        iters (int): Maximum number of bisection iterations
        tol (double): Convergence tolerance of iterations

    Returns:

        Argument at which loss function is bisected
    """

    eta1 = start_interval[0]
    eta2 = start_interval[1]
    obj3 = function(eta2)
    if np.allclose(eta1, eta2, 1e-12):
        return eta1
    if np.sign(function(eta1)) == np.sign(function(eta2)):
        print("[Bisection Method] There is no root in this range.")
        val = np.argmin(np.abs([eta1, eta2]))
        return [eta1, eta2][val]
    for i in range(int(iters)):
        obj1 = function(eta1)
        eta3 = (eta2 + eta1) * 0.5
        obj3 = function(eta3)
        if np.abs(eta1 - eta3) / np.abs(eta3) < tol:
            # if np.abs(obj3) < tol:
            return eta3
        if np.sign(obj1) == np.sign(obj3):
            eta1 = eta3
        else:
            eta2 = eta3
    print("Did not converge... ", obj3)
    return eta3


def create_local_credible_interval(
    x_sol, region_size, function, bound, iters, tol, bottom, top, verbose=0.
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

    Returns:

        Upper limit, lower limit, super-pixel mean
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
                    return - bound + function(
                        x_sol * (1.0 - mask) + (x_sum + eta) * mask
                    ) 

                error_p[i, j] = bisection_method(obj, [0, top], iters, tol) + x_sum

                def obj(eta):
                    return - bound + function(
                        x_sol * (1.0 - mask) + (x_sum - eta) * mask
                    ) 

                error_m[i, j] = -bisection_method(obj, [0, -bottom], iters, tol) + x_sum
                
                if verbose > 0.:
                    print(
                        "[Credible Interval] (%s, %s) has interval (%s, %s) with sum %s"%(
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
            
            if verbose > 0.:
                print(
                    "[Credible Interval] %s has interval (%s, %s) with sum %s"%(
                        i,
                        error_m[i],
                        error_p[i],
                        x_sum,
                    )
                )
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


# def create_superpixel_map(x_sol, region_size):
#     """Bisection method for finding credible interval."""
#
#!     region = np.zeros(x_sol.shape)
#     if len(x_sol.shape) > 1:
#!         region[:region_size, :region_size] = 1.0
#         dsizey, dsizex = int(x_sol.shape[0] / region_size), int(
#             x_sol.shape[1] / region_size
#         )
#         mean = np.zeros((dsizey, dsizex))
#         for i in range(dsizey):
#             for j in range(dsizex):
#                 mask = np.roll(
#                     np.roll(region, shift=i * region_size, axis=0),
#                     shift=j * region_size,
#                     axis=1,
#                 )
#                 x_sum = np.nansum(np.ravel(x_sol[(mask.astype(bool))]))
#                 mean[i, j] = x_sum
#     else:
#!         region[:region_size] = 1.0
#         dsizey = int(x_sol.shape[0] / region_size)
#         mean = np.zeros((dsizey))
#         for i in range(dsizey):
#             mask = np.roll(region, shift=i * region_size, axis=0)
#             x_sum = np.nansum(np.ravel(x_sol[(mask.astype(bool))]))
#             mean[i] = x_sum
#     return mean
