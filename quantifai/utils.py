"""
Utils and helper functions

Some of these functions come from https://github.com/MI2G/sampling-tutorials

Authors and credit
* Dobson, Paul [pdobson@ed.ac.uk](pdobson@ed.ac.uk)
* Kemajou, Mbakam Charlesquin [cmk2000@hw.ac.uk](cmk2000@hw.ac.uk)
* Klatzer, Teresa [t.klatzer@sms.ed.ac.uk](t.klatzer@sms.ed.ac.uk)
* Melidonis, Savvas [sm2041@hw.ac.uk](sm2041@hw.ac.uk)
"""

import numpy as np
import torch
import arviz
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.graphics.tsaplots import plot_acf


def to_numpy(z):
    """
    Converts a PyTorch tensor or variable to a numpy array on the CPU.

    Args:
        z (torch.Tensor): The input tensor or variable.

    Returns:
        numpy.ndarray: The converted numpy array.

    """
    return z.detach().cpu().squeeze().numpy()


def to_tensor(z, device="cuda", dtype=torch.float):
    """
    Converts a numpy array to a PyTorch tensor on a specified device.

    Args:
        z (numpy.ndarray): The input numpy array.
        device (str): The device to place the resulting tensor on (default is 'cuda').
        dtype (torch.dtype): The datatype of the resulting tensor (default is torch.float).

    Returns:
        torch.Tensor: The converted PyTorch tensor, reshaped to have 1 batch dimension.

    """
    return torch.tensor(z, device=device, dtype=dtype, requires_grad=False).reshape(
        (1, 1) + z.shape
    )


def eval_snr(x, x_est):
    """
    Calculates the Signal-to-Noise Ratio (SNR) in decibels (dB) between two signals.

    Args:
        x (np.ndarray): The original signal.
        x_est (np.ndarray): The estimated or reconstructed signal.

    Returns:
        float: The SNR between `x` and `x_est` in dB.

    Raises:
        ValueError: If `x` and `x_est` are not of the same shape, an error will be
            raised as the signals must be of the same length to calculate SNR.
    """
    if x.shape != x_est.shape:
        raise ValueError(
            "The shapes of `x` and `x_est` must be the same to calculate SNR."
        )
    if np.array_equal(x, x_est):
        return 0
    num = np.sqrt(np.sum(np.abs(x) ** 2))
    den = np.sqrt(np.sum(np.abs(x - x_est) ** 2))
    return round(20 * np.log10(num / den), 2)


def nrmse(x, y):
    """Compute the normalized root mean square error (NRMSE)"""
    x_np = to_numpy(x)
    return np.linalg.norm(x_np - to_numpy(y), "fro") / np.linalg.norm(x_np, "fro")


def clip_matrix(mat, low_val=None, high_val=None):
    """
    Clips the values of a numpy matrix to within a specified range.

    Args:
        mat (np.ndarray): The input matrix to be clipped.
        low_val (float or None): The minimum value to clip the matrix to.
            If None, the matrix will not be clipped at the lower end.
        high_val (float or None): The maximum value to clip the matrix to.
            If None, the matrix will not be clipped at the upper end.

    Returns:
        np.ndarray: The clipped matrix.

    """
    if low_val is not None:
        mat[mat < low_val] = low_val
    if high_val is not None:
        mat[mat >= high_val] = high_val
    return mat


def blur_operators(kernel_len, size, type_blur, device):
    """
    Generates the forward and backward blur operators for a given blur kernel type and size.

    Args:
        kernel_len (tuple): The length of the blur kernel in the x and y directions.
        size (tuple): The size of the image being blurred in the x and y directions.
        type_blur (str): The type of blur kernel to use. Only 'uniform' is currently supported.
        device (str): The device to place the resulting operators on (e.g. 'cuda', 'cpu').

    Returns:
        Tuple of functions and scalar: A tuple containing the forward operator `A`,
            the backward operator `AT`, and the normalization constant `AAT_norm`.

    Raises:
        NotImplementedError: If `type_blur` is not 'uniform', an error will be raised
            as only uniform blur is currently supported.

    """
    nx = size[0]
    ny = size[1]
    if type_blur == "uniform":
        h = torch.zeros(nx, ny).to(device)
        lx = kernel_len[0]
        ly = kernel_len[1]
        h[0:lx, 0:ly] = 1 / (lx * ly)
        c = np.ceil((np.array([ly, lx]) - 1) / 2).astype("int64")
    else:
        print("Write more code!")
        raise NotImplementedError

    H_FFT = torch.fft.fft2(torch.roll(h, shifts=(-c[0], -c[1]), dims=(0, 1)))
    HC_FFT = torch.conj(H_FFT)

    # A forward operator
    A = lambda x: torch.fft.ifft2(
        torch.multiply(H_FFT, torch.fft.fft2(x[0, 0]))
    ).real.reshape(x.shape)
    # A backward operator
    AT = lambda x: torch.fft.ifft2(
        torch.multiply(HC_FFT, torch.fft.fft2(x[0, 0]))
    ).real.reshape(x.shape)

    AAT_norm = max_eigenval(A, AT, size, 1e-4, int(1e4), 0, device=device)

    return A, AT, AAT_norm


def max_eigenval(
    A,
    At,
    im_shape,
    tol=1e-4,
    max_iter=np.int64(1e4),
    verbose=0,
    device=torch.device("cpu"),
):
    """Computes the maximum eigen value of the compund operator AtA.

    Follows X. Cai article's areas.

    Args:
        A (Callable): Radio image name
        At (Callable): if the area contains phyisical information
        im_shape (list, tuple or np.ndarray): Image shape where `len(im_shape)=2`
        tol (float): Algorithm's tolerance
        max_iter (int): Max number of iterations
        verbose (float): If verbose>0, verbose mode is activated
        device (torch.device): Torch's device

    Returns:
        val (float): max eigenvalue of the AtA operator
    """
    with torch.no_grad():
        x = torch.normal(mean=0, std=1, size=im_shape, device=device)
        x = x / torch.norm(torch.ravel(x), 2)
        init_val = 1

        for k in range(max_iter):
            y = A(x)
            x = At(y)
            val = torch.norm(torch.ravel(x), 2)
            rel_var = torch.abs(val - init_val) / init_val
            if verbose > 1:
                print("Iter = {}, norm = {}", k, val)

            if rel_var < tol:
                break

            init_val = val
            x = x / val

        if verbose > 0:
            print("Norm = {}", val)

        return val


class welford:
    """
    Welford's algorithm for calculating mean and variance
    Collection of summary statistics of the Markov chain iterates

    call update() when generating a new sample
    call get_mean() or get_var() to obtain the current mean or variance

    https://doi.org/10.2307/1266577
    """

    def __init__(self, x):
        self.k = 1
        self.M = x.clone()
        self.S = 0

    def update(self, x):
        self.k += 1
        Mnext = self.M + (x - self.M) / self.k
        self.S += (x - self.M) * (x - Mnext)
        self.M = Mnext

    def get_mean(self):
        return self.M

    def get_var(self):
        # when the number of samples is 1, we divide by self.k
        if self.k < 2:
            return self.S / (self.k)
        else:
            return self.S / (self.k - 1)


def ESS(arr):
    """Calculate the effective sample size (ESS) of an array.

    This function uses the ArviZ package to calculate the ESS.

    Args:
        arr (np.ndarray): The array for which to calculate the ESS, usually
            the trace of the Markov chain.

    Returns:
        float: The effective sample size of the array.
    """
    ess = arviz.ess(arr)
    return ess


def autocor_plots(X_chain, var, method_str, nLags=100, save_path=None):
    """Plots autocorrelation functions of traces of a Markov Chain.

    Args:
        X_chain (np.ndarray):  The Markov chain with shape `(num_samples, num_variables)`.
        var (np.ndarray): The variance in the Fourier domain.
        method_str (str): The name of the algorithm/method used to obtain the samples.
        nLags (int): Number of lags.
        save_path (str or None): Path to save the figure. If None, the figure is not saved.

    Raises:
        ValueError: if the number of lags is greater than the number of samples.

    Returns:
        None
    """

    # Checking whether the samples size is greater than the number of lags considered.
    if X_chain.shape[0] < nLags:
        raise ValueError(f"nLags must be smaller than the number of samples!")

    # Convert tensor to numpy array
    if torch.is_tensor(X_chain):
        X_chain = X_chain.detach().cpu().numpy()

    # Vectorise the variance in the Fourier domain
    var_fft = var.reshape(-1, 1)

    # Vectorise the Markov chain
    X_chain_vec = X_chain.reshape(len(X_chain), -1)

    # Variance of the Markov not in the spatial
    var_sp = np.var(X_chain_vec, axis=0)

    # Trace of the Markov chain with the maximum variance
    trace_elem_max_variance = X_chain_vec[:, np.argmax(var_sp)]

    # Trace of the Markov chain with the minimum variance
    trace_elem_min_variance = X_chain_vec[:, np.argmin(var_sp)]

    # Trace of the Markov chain with median variance
    ind_medium = np.argsort(var_sp)[len(var_sp) // 2]
    trace_elem_median_variance = X_chain_vec[:, ind_medium]

    # Effective sample size
    e_slow = ESS(trace_elem_max_variance.reshape(-1))
    e_fast = ESS(trace_elem_min_variance.reshape(-1))
    e_med = ESS(trace_elem_median_variance.reshape(-1))

    # Generate the autocorrelation function for these three traces: lower, medium and faster.
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_acf(
        trace_elem_median_variance,
        ax=ax,
        label=f"Median-speed, ESS = {e_med: .2f}",
        alpha=None,
        lags=nLags,
    )
    plot_acf(
        trace_elem_max_variance,
        ax=ax,
        label=f"Slowest, ESS = {e_slow: .2f}",
        alpha=None,
        lags=nLags,
    )
    plot_acf(
        trace_elem_min_variance,
        ax=ax,
        label=f"Fastest, ESS = {e_fast: .2f}",
        alpha=None,
        lags=nLags,
    )
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[1::2]
    labels = labels[1::2]
    ax.set_title(method_str)
    ax.set_ylabel("ACF")
    ax.set_xlabel("lags")
    ax.set_ylim([-1.1, 1.3])
    ax.legend(handles=handles, labels=labels, loc="best", shadow=True, numpoints=1)

    if save_path is not None:
        plt.savefig(save_path)


def plot_summaries(
    x_ground_truth,
    x_dirty,
    post_meanvar,
    post_meanvar_absfourier,
    cmap="gray",
    save_path=None,
):
    """Plot summaries of the sampling results.

    Args:
        x_ground_truth (np.ndarray): Ground truth image.
        x_dirty (np.ndarray): Dirty image.
        post_meanvar (welford instance): Instance of welford with torch variables
            saving the MC samples.
        post_meanvar_absfourier (welford instance): Instance of welford with torch
            variables saving the absolute values of Fourier coefficients of MC samples.
        save_path (str): Path to save the figure. If None, the figure is not saved.

    Returns:
        None
    """

    post_mean_numpy = post_meanvar.get_mean().detach().cpu().squeeze().numpy()
    post_var_numpy = post_meanvar.get_var().detach().cpu().squeeze().numpy()

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 10))
    fig.tight_layout(pad=0.01)

    # Ground truth
    im = axes[0, 0].imshow(x_ground_truth, cmap=cmap)
    axes[0, 0].set_title("Ground truth image")
    axes[0, 0].axis("off")
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    # Dirty image
    im = axes[0, 1].imshow(x_dirty, cmap=cmap)
    axes[0, 1].set_title("Dirty image")
    axes[0, 1].axis("off")
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    # MMSE
    im = axes[0, 2].imshow(post_mean_numpy, cmap=cmap)
    axes[0, 2].set_title("x - posterior mean")
    axes[0, 2].axis("off")
    divider = make_axes_locatable(axes[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    # Variance
    im = axes[0, 3].imshow(post_var_numpy, cmap=cmap)
    axes[0, 3].set_title("x - posterior variance")
    axes[0, 3].axis("off")
    divider = make_axes_locatable(axes[0, 3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    # MMSE / Var
    im = axes[1, 0].imshow(
        post_mean_numpy
        / np.sqrt(post_meanvar.get_var().detach().cpu().squeeze().numpy()),
        cmap=cmap,
    )
    axes[1, 0].set_title("x - posterior mean/posterior SD")
    axes[1, 0].axis("off")
    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    # Var / MMSE
    im = axes[1, 1].imshow(np.sqrt(post_var_numpy) / post_mean_numpy, cmap=cmap)
    axes[1, 1].set_title("x - Coefs of variation")
    axes[1, 1].axis("off")
    divider = make_axes_locatable(axes[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    # Mean Fourier coefs
    im = axes[1, 2].imshow(
        torch.log(post_meanvar_absfourier.get_mean()).detach().cpu().squeeze().numpy()
    )
    axes[1, 2].set_title("Mean coefs (log-scale)")
    axes[1, 2].axis("off")
    divider = make_axes_locatable(axes[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    # Variance Fourier coefs
    im = axes[1, 3].imshow(
        torch.log(post_meanvar_absfourier.get_var()).detach().cpu().squeeze().numpy()
    )
    axes[1, 3].set_title("Var coefs (log-scale)")
    axes[1, 3].axis("off")
    divider = make_axes_locatable(axes[1, 3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    if save_path is not None:
        plt.savefig(save_path)
