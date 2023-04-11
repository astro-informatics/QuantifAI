
import torch
import math
import numpy as np
from typing import Callable

def ULA_kernel(
    X: torch.Tensor,
    delta: float,
    grad_likelihood_prior: Callable
) -> torch.Tensor:
    """ULA sampling algorithm kernel

    Args:

        X (torch.Tensor): Tensor to update
        delta (float): Step size for the ULA algorithm
        grad_likelihood_prior (function): drift term or gradient of the likelihood and prior terms

    Returns:
        torch.Tensor: New generated sample
    """
    return X - delta * grad_likelihood_prior(X) + math.sqrt(2*delta) * torch.randn_like(X)


def MYULA_kernel(
    X: torch.Tensor,
    delta: float,
    lmbd: float,
    grad_likelihood: Callable,
    prox_prior: Callable,
    op_drift= lambda _x : torch.real(_x)
) -> torch.Tensor:
    """ULA sampling algorithm kernel

    Args:

        X (torch.Tensor): Tensor to update
        delta (float): Step size for the MYULA algorithm
        lmbd (float): Moreau-Yosida envelope parameter
        grad_likelihood (function): gradient of the likelihood
        prox_prior (function): prox of the non-smooth prior
        op_drift (function): operator to apply to the drift term.
            Defaults to the real projection.

    Returns:
        torch.Tensor: New generated sample
    """
    return (1. - (delta/lmbd)) * torch.clone(X) + op_drift(
        - delta * grad_likelihood(torch.clone(X)) + (delta/lmbd) * prox_prior(X, lmbd)
    ) + math.sqrt(2*delta) * torch.randn_like(X)


def SKROCK_kernel(
    X: torch.Tensor,
    Lipschitz_U: float,
    nStages: int,
    eta: float,
    dt_perc: float,
    grad_likelihood_prior: Callable
) -> torch.Tensor:
    """SKROCK sampling algorithm kernel

    Args:

        X (torch.Tensor): Tensor to update
        Lipschitz_U (float): Lipschitz constant of the likelihood and prior terms
        nStages (float): Number of gradient evaluations to store and use for the update
        eta (int): Variable appearing in the max step-size calculation
        dt_perc (float): Percentage of the step-size to be used
        grad_likelihood_prior (function): drift term or gradient of the likelihood and prior terms

    Returns:
        Xts (torch.Tensor): New generated sample
    """
    # SK-ROCK parameters

    # First kind Chebyshev function
    T_s = lambda s, x : np.cosh(s * np.arccosh(x))

    # First derivative Chebyshev polynomial first kind
    T_prime_s = lambda s, x : s * np.sinh(s * np.arccosh(x)) / np.sqrt(x**2 -1)

    # computing SK-ROCK stepsize given a number of stages
    # and parameters needed in the algorithm
    denNStag = (2 - (4/3) * eta)
    rhoSKROCK = ((nStages - 0.5)**2) * denNStag - 1.5 # stiffness ratio
    dtSKROCK = dt_perc * rhoSKROCK / Lipschitz_U # step-size

    w0 = 1 + eta / (nStages**2) # parameter \omega_0
    w1 = T_s(nStages,w0) / T_prime_s(nStages,w0) # parameter \omega_1
    mu1 = w1/w0 # parameter \mu_1
    nu1 = nStages * (w1/2) # parameter \nu_1
    kappa1 = nStages * (w1/w0) # parameter \kappa_1

    # Sampling the variable X (SKROCK)
    Q = math.sqrt(2 * dtSKROCK) * torch.randn_like(X) # diffusion term

    # SKROCK

    # SKROCK first internal iteration (s=1)
    XtsMinus2 = X.clone()
    Xts= X.clone() - mu1 * dtSKROCK * grad_likelihood_prior(X + nu1 * Q) + kappa1 * Q

    for js in range(2, nStages + 1): # s=2,...,nStages SK-ROCK internal iterations
        XprevSMinus2 = Xts.clone()
        mu = 2 * w1 * T_s(js - 1, w0) / T_s(js, w0) # parameter \mu_js
        nu = 2 * w0 * T_s(js - 1, w0) / T_s(js, w0) # parameter \nu_js
        kappa = 1 - nu # parameter \kappa_js
        Xts = -mu * dtSKROCK * grad_likelihood_prior(Xts) + nu * Xts + kappa * XtsMinus2
        XtsMinus2 = XprevSMinus2

    return Xts # new sample produced by the SK-ROCK algorithm


