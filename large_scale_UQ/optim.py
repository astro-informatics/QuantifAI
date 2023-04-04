
import numpy as np
import time
import torch
from functools import partial
from large_scale_UQ import empty as Empty
from large_scale_UQ.operators import L1Norm_torch

def FB_torch(
        x_init,
        options=None,
        g=None,
        f=None,
        h=None,
        alpha=1,
        tau=1,
        viewer=None
    ):
    """Evaluates the base forward backward optimisation

    Note that currently this only supports real positive semi-definite
    fields.

    Args:

        x_init (np.ndarray): First estimate solution
        options (dict): Python dictionary of optimisation configuration parameters
        g (Grad Class): Unconstrained data-fidelity class
        f (Prox Class): Reality constraint
        h (Prox/AI Class): Proximal or Learnt regularisation constraint
        alpha (float): regularisation paremeter / step-size.
        tau (float): custom weighting of proximal operator
        viewer (function): Plotting function for real-time viewing (must accept: x, iteration)
    """
    if f is None:
        f = Empty.EmptyProx()
    if g is None:
        g = Empty.EmptyGrad()
    if h is None:
        h = Empty.EmptyProx()

    if options is None:
        options = {"tol": 1e-4, "iter": 500, "update_iter": 100, "record_iters": False}

    # algorithmic parameters
    tol = options["tol"]
    max_iter = options["iter"]
    update_iter = options["update_iter"]
    record_iters = options["record_iters"]

    # initialization
    x = torch.clone(x_init)

    print("Running Base Forward Backward")
    timing = np.zeros(max_iter)
    criter = np.zeros(max_iter)

    # Set subtraction operation
    _op_sub = lambda _x1, _x2 : _x1 - _x2

    if isinstance(h, L1Norm_torch):
        if h.num_wavs > 0:
            op_sub = partial(h._op_to_two_coeffs, op=_op_sub)
    else: 
        op_sub = _op_sub

    # algorithm loop
    for it in range(0, max_iter):

        t = time.time()
        # forward step
        x_old = torch.clone(x)
        x = x - alpha * g.grad(x)
        x = f.prox(x, tau)

        # backward step
        u = h.dir_op(x)
        u2 = h.dir_op(torch.clone(x))
        x = x + h.adj_op(op_sub(h.prox(u, tau), u2))

        # time and criterion
        if record_iters:
            timing[it] = time.time() - t
            criter[it] = f.fun(x) + g.fun(x) + h.fun(h.dir_op(x))

        if torch.allclose(x, torch.tensor(0., dtype=x.dtype)):
            x = x_old
            print("[Forward Backward] converged to 0 in %d iterations"%(it))
            break
        # stopping rule
        res = (torch.norm(x - x_old) / torch.norm(x_old)).item()
        if it > 10:
            if res < tol:
                print("[Forward Backward] converged in %d iterations"%(it))
                break
        if update_iter >= 0:
            if it % update_iter == 0:
                print(
                    "[Forward Backward] %d out of %d iterations, tol = %.2e"%(
                    it, max_iter, res,
                    )
                )
                if viewer is not None:
                    viewer(x, it)


    criter = criter[0 : it + 1]
    timing = np.cumsum(timing[0 : it + 1])
    solution = x
    diagnostics = {
        "max_iter": it,
        "times": timing,
        "Obj_vals": criter,
        "x": x,
    }
    return solution, diagnostics

