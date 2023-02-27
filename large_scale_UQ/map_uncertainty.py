import numpy as np
import logging

logger = logging.getLogger("Optimus Primal")


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
        logger.info("[Bisection Method] There is no root in this range.")
        val = np.argmin(np.abs([eta1, eta2]))
        return [eta1, eta2][val]
    for i in range(int(iters)):
        obj1 = function(eta1)
        eta3 = (eta2 + eta1) * 0.5
        obj3 = function(eta3)
        if np.abs(eta1 - eta3) / np.abs(eta3) < tol:
            if np.abs(obj3) < tol:
                return eta3
        if np.sign(obj1) == np.sign(obj3):
            eta1 = eta3
        else:
            eta2 = eta3
    print("Did not converge... ", obj3)
    return eta3


def create_local_credible_interval(
    x_sol, region_size, function, bound, iters, tol, bottom, top
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
    logger.info("Calculating credible interval for %s superpxiels.", region.shape)
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
                logger.info(
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
            logger.info(
                "[Credible Interval] %s has interval (%s, %s) with sum %s",
                i,
                error_m[i],
                error_p[i],
                x_sum,
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
    logger.info("Calculating credible interval for %s superpxiels.", region.shape)
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
                logger.info(
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
            logger.info(
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
