import numpy as np
import math
import torch
from quantifai.utils import max_eigenval
from quantifai.empty import Identity
import ptwt


class MaskedFourier(object):
    """
    Masked fourier sensing operator i.e. MRI/Radio imaging.
    """

    def __init__(
        self, dim, ratio=0.5, mask=None, norm=None, framework="numpy", device="cpu"
    ):
        """Initialises the masked fourier sensing operator.

        Args:
            dim (int): Dimension of square pixel-space image.
            ratio (float): Fraction of measurements observed.
            norm (str): FFT normalization mode. Options are `forward`, `backward` or `norm`.
            framework (str): Framework type. Options are `numpy` or `pytorch`.
            device (str): device for the `pytorch` framework.
        """
        self.norm = norm
        self.dim = dim
        self.mask = mask
        self.framework = framework
        self.device = device
        self.shape = (dim, dim)
        # If the mask is not defined, initialise a random one
        if mask is None:
            mask = np.full(dim**2, False)
            mask[: int(ratio * dim**2)] = True
            np.random.shuffle(mask)
            self.mask = mask.reshape(self.shape)
        if self.framework == "pytorch":
            self.mask = torch.tensor(np.copy(mask), device=device).reshape(
                (1, 1) + self.shape
            )

        if self.framework == "numpy":
            self.dir_op = self._numpy_dir_op
            self.adj_op = self._numpy_adj_op
        elif self.framework == "pytorch":
            self.dir_op = self._torch_dir_op
            self.adj_op = self._torch_adj_op

    def set_mask(self, new_mask):
        """Set new mask taking care of the framework."""
        if isinstance(new_mask, np.ndarray):
            if self.framework == "numpy":
                self.mask = new_mask
            elif self.framework == "pytorch":
                self.mask = torch.tensor(np.copy(new_mask), device=self.device).reshape(
                    (1, 1) + new_mask.shape
                )
        elif isinstance(new_mask, torch.Tensor):
            if self.framework == "numpy":
                self.mask = new_mask.detach().cpu().squeeze().numpy()
            elif self.framework == "pytorch":
                self.mask = new_mask

    def _torch_dir_op(self, x):
        """Computes the forward operator of the class.

        Compute FFT and then mask Fourier coefficients.
        Using the `pytorch` framework.

        Args:
            x (torch.Tensor): Array to apply FFT and mask. Torch shape: (1, 1, dim, dim)

        Returns:
            torch.Tensor: tensor of Fourier coefficients. Same shape as input.
        """

        return torch.mul(torch.fft.fft2(x, norm=self.norm), self.mask)

    def _torch_adj_op(self, x):
        """Computes the forward adjoint operator of the class.

        Compute the inverse FFT.

        Args:
            x (torch.Tensor): tensor of Fourier coefficients. Same shape as output.

        Returns:
            torch.Tensor: Output array. Torch shape: (1, 1, dim, dim)
        """

        return torch.fft.ifft2(x, norm=self.norm)

    def _numpy_dir_op(self, x):
        """Computes the forward operator of the class.

        Using the `numpy` framework.

        Args:
            x (np.ndarray): Array to apply FFT and mask.

        Returns:
            np.ndarray: array of coefficients
        """
        out = np.fft.fft2(x, norm=self.norm)
        return self._numpy_mask(out)

    def _numpy_adj_op(self, x):
        """Computes the forward adjoint operator of the class.

        Using the `numpy` framework.

        Args:
            x (np.ndarray): Vector to apply identity to.

        Returns:
            np.ndarray: array of coefficients
        """
        out = self._numpy_mask_adjoint(x)
        return np.fft.ifft2(out, norm=self.norm)

    def _numpy_mask(self, x):
        """Applies observational mask to image.

        Using the `numpy` framework.

        Args:
            x (np.ndarray): Vector to apply mask to.

        Returns:
            np.ndarray: slice of masked coefficients
        """
        return x[self.mask]

    def _numpy_mask_adjoint(self, x):
        """Applies adjoint of observational mask to image.

        Using the `numpy` framework.

        Args:
            x (np.ndarray): Vector to apply adjoint mask to.

        Returns:
            np.ndarray: Projection of masked coefficients onto image.
        """
        xx = np.zeros(self.shape, dtype=complex)
        xx[self.mask] = x
        return xx


class MaskedFourier_numpy:
    """
    Masked fourier sensing operator i.e. MRI/Radio imaging.
    """

    def __init__(
        self,
        dim,
        ratio=0.5,
        mask=None,
        norm=None,
    ):
        """Initialises the masked fourier sensing operator.

        Args:
            dim (int): Dimension of square pixel-space image.
            ratio (float): Fraction of measurements observed.
            norm (str): FFT normalization mode. Options are `forward`, `backward` or `norm`.
        """
        self.norm = norm
        self.dim = dim
        self.mask = mask
        self.shape = (dim, dim)
        # If the mask is not defined, initialise a random one
        if mask is None:
            mask = np.full(dim**2, False)
            mask[: int(ratio * dim**2)] = True
            np.random.shuffle(mask)
            self.mask = mask.reshape(self.shape)

        # Set the operations for the numpy framework
        self.dir_op = self._numpy_dir_op
        self.adj_op = self._numpy_adj_op

    def set_mask(self, new_mask):
        """Set new mask taking care of the framework."""
        if isinstance(new_mask, np.ndarray):
            self.mask = new_mask
        elif isinstance(new_mask, torch.Tensor):
            self.mask = new_mask.detach().cpu().squeeze().numpy()

    def _numpy_dir_op(self, x):
        """Computes the forward operator of the class.

        Using the `numpy` framework.

        Args:
            x (np.ndarray): Array to apply FFT and mask.

        Returns:
            np.ndarray: array of coefficients
        """
        out = np.fft.fft2(x, norm=self.norm)
        return self._numpy_mask(out)

    def _numpy_adj_op(self, x):
        """Computes the forward adjoint operator of the class.

        Using the `numpy` framework.

        Args:
            x (np.ndarray): Vector to apply identity to.

        Returns:
            np.ndarray: array of coefficients
        """
        out = self._numpy_mask_adjoint(x)
        return np.fft.ifft2(out, norm=self.norm)

    def _numpy_mask(self, x):
        """Applies observational mask to image.

        Using the `numpy` framework.

        Args:
            x (np.ndarray): Vector to apply mask to.

        Returns:
            np.ndarray: slice of masked coefficients
        """
        return x[self.mask]

    def _numpy_mask_adjoint(self, x):
        """Applies adjoint of observational mask to image.

        Using the `numpy` framework.

        Args:
            x (np.ndarray): Vector to apply adjoint mask to.

        Returns:
            np.ndarray: Projection of masked coefficients onto image.
        """
        xx = np.zeros(self.shape, dtype=complex)
        xx[self.mask] = x
        return xx


class MaskedFourier_torch(torch.nn.Module):
    """
    Masked fourier sensing operator i.e. MRI/Radio imaging.
    """

    def __init__(self, shape, ratio=0.5, mask=None, norm=None, device="cpu"):
        """Initialises the masked fourier sensing operator.

        Args:
            shape (list, tuple or int): Dimensions of pixel-space image.
            ratio (float): Fraction of measurements observed.
            norm (str): FFT normalization mode. Options are `forward`, `backward` or `norm`.
            device (str): device for the `pytorch` framework.
        """
        super().__init__()
        self.norm = norm
        self.ratio = ratio
        self.device = device
        if type(shape) is int:
            self.shape = (shape, shape)
        elif len(shape) == 2:
            self.shape = shape
        else:
            raise ValueError("Shape should be an int or array (or tuple) of length 2.")
        self.mask = mask
        # If the mask is not defined, initialise a random one
        if mask is None:
            self.init_mask()
        else:
            # Check the channel dimensions for the pytorch framework
            if self.mask.shape != (1, 1, self.shape[0], self.shape[1]):
                self.mask.reshape((1, 1) + self.shape)
            # Set init mask
            self.set_mask(self.mask)
        # Set the operations from the torch framework
        self.dir_op = self._torch_dir_op
        self.adj_op = self._torch_adj_op
        # Define pytorch module attributes
        self.training = False
        self.requires_grad_(requires_grad=False)
        self.forward = self._torch_dir_op

    def init_mask(self):
        """Initialise random mask."""
        mask = np.full(self.shape[0] * self.shape[1], False)
        mask[: int(self.ratio * self.shape[0] * self.shape[1])] = True
        np.random.shuffle(mask)
        self.mask = torch.tensor(
            np.copy(mask.reshape(self.shape)), device=self.device
        ).reshape((1, 1) + self.shape)

    def set_mask(self, new_mask):
        """Set new mask."""
        if isinstance(new_mask, np.ndarray):
            self.mask = torch.tensor(np.copy(new_mask), device=self.device).reshape(
                (1, 1) + self.shape
            )
        elif isinstance(new_mask, torch.Tensor):
            self.mask = new_mask.reshape((1, 1) + self.shape)

    def _torch_dir_op(self, x):
        """Computes the forward operator of the class.

        Compute FFT and then mask Fourier coefficients.
        Using the `pytorch` framework.

        Args:
            x (torch.Tensor): Array to apply FFT and mask. Torch shape: (1, 1, dim, dim)

        Returns:
            torch.Tensor: tensor of Fourier coefficients. Same shape as input.
        """

        return torch.mul(torch.fft.fft2(x, norm=self.norm), self.mask)

    def _torch_adj_op(self, x):
        """Computes the forward adjoint operator of the class.

        Compute the inverse FFT.

        Args:
            x (torch.Tensor): tensor of Fourier coefficients. Same shape as output.

        Returns:
            torch.Tensor: Output array. Torch shape: (1, 1, dim, dim)
        """

        return torch.fft.ifft2(x, norm=self.norm)


class L2Norm_torch(torch.nn.Module):
    """This class computes the gradient operator of the l2 norm function.

                        f(x) = ||y - Phi x||^2/2/sigma^2

    When the input 'x' is an array. 'y' is a data vector, `sigma` is a scalar uncertainty
    """

    def __init__(self, sigma, data, Phi=None):
        """Initialises the l2_norm class

        Args:

            sigma (double): Noise standard deviation
            data (np.ndarray): Observed data
            Phi (Linear operator): Sensing operator

        Raises:

            ValueError: Raised when noise std is not positive semi-definite

        """
        super().__init__()
        if np.any(sigma <= 0):
            raise ValueError("'sigma' must be positive")
        # Set parameters and data
        self.sigma = sigma
        self.data = data
        if Phi is None:
            self.Phi = Identity()
            # Compute Lipschitz constant
            self.beta = 1.0 / sigma**2
        else:
            self.Phi = Phi
            self._compute_lip_constant()

        # Define pytorch module attributes
        self.training = False
        self.requires_grad_(requires_grad=False)
        # Define forward operation as the gradient computation
        self.forward = self.grad

    def _compute_lip_constant(self):
        """Compute Lipschitz constant."""
        A = lambda _x: self.Phi.dir_op(_x)
        At = lambda _x: self.Phi.adj_op(_x)

        max_val = max_eigenval(
            A=A,
            At=At,
            im_shape=torch.squeeze(self.data).shape,
            tol=1e-4,
            max_iter=int(1e4),
            verbose=0,
            device=self.data.device,
        )
        # Store Lipschitz constant
        self.beta = max_val.item() / self.sigma**2

    def grad(self, x, sigma=None, sigma2=None):
        """Gradient of the l2_norm class with respect to the data

        Args:

            x (torch.Tensor): Data estimate
            sigma (float): Noise standard deviation
            sigma2 (float): Noise variance

        Returns:

            Gradient of the l2_norm expression

        """
        if sigma is None:
            if sigma2 is None:
                sigma = self.sigma
            else:
                sigma = math.sqrt(sigma2)

        return -self.Phi.adj_op((self.data - self.Phi.dir_op(x))) / (sigma**2)

    def grad_sigma(self, x, sigma=None):
        """Gradient of the l2_norm class with respect to sigma

        Args:

            x (torch.Tensor): Data estimate
            sigma (torch.Tensor): Noise std dev value

        Returns:

            Gradient of the l2_norm expression

        """
        if sigma is None:
            sigma = self.sigma
        return -torch.sum(torch.abs(self.data - self.Phi.dir_op(x)) ** 2.0) / (
            sigma**3
        )

    def grad_sigma2(self, x, sigma2=None):
        """Gradient of the l2_norm class with respect to sigma**2

        Args:

            x (torch.Tensor): Data estimate
            sigma2 (torch.Tensor): Noise std dev value

        Returns:

            Gradient of the l2_norm expression

        """
        if sigma2 is None:
            sigma2 = self.sigma**2
        return -torch.sum(torch.abs(self.data - self.Phi.dir_op(x)) ** 2.0) / (
            2 * sigma2**2
        )

    def fun(self, x, sigma=None, sigma2=None):
        """Evaluates the l2_norm class

        Args:

            x (torch.Tensor): Data estimate
            sigma (float): Noise std dev value
            sigma2 (float): Noise variance

        Returns:

            Computes the l2_norm loss

        """
        if sigma is None:
            if sigma2 is None:
                sigma = self.sigma
            else:
                sigma = math.sqrt(sigma2)

        return torch.sum(torch.abs(self.data - self.Phi.dir_op(x)) ** 2.0) / (
            2 * sigma**2
        )


class Wavelets_torch(torch.nn.Module):
    """
    Constructs a linear operator for abstract Daubechies Wavelets
    """

    def __init__(self, wav, levels, mode="periodic"):
        """Initialises Daubechies Wavelet linear operator class

        Args:

            wav (string): Wavelet type (see https://tinyurl.com/5n7wzpmb)
            levels (int): Wavelet levels (scales) to consider
            mode (str): Wavelet signal extension mode

        Raises:

            ValueError: Raised when levels are not positive definite
            ValueError: Raised if the wavelet type is not a string

        """
        super().__init__()
        if np.any(levels <= 0):
            raise ValueError("'levels' must be positive")
        if not isinstance(wav, str):
            raise ValueError("'wav' must be a string")
        self.wav = wav
        self.levels = np.int64(levels)
        self.mode = mode

        self.adj_op(self.dir_op(torch.ones((1, 64, 64))))

    def dir_op(self, x):
        """Evaluates the forward abstract wavelet transform of x

        Args:

            x (torch.Tensor): Array to wavelet transform. Can be [batch, H, W] or [H, W],
                but it will raise an error if used with [batch, channels, H, W].

        Returns:

            coeffs (List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                Wavelet decomposition coefficients

        Raises:

            ValueError: Raised when the shape of x is not even in every dimension
        """
        # TODO: Add special case for wavelet `self` 
        if x.dim() >= 4:
            return ptwt.wavedec2(
                x.squeeze(1), wavelet=self.wav, level=self.levels, mode=self.mode
            )
        else:
            return ptwt.wavedec2(x, wavelet=self.wav, level=self.levels, mode=self.mode)

    def adj_op(self, coeffs):
        """Evaluates the forward adjoint abstract wavelet transform of x

        Args:

            coeffs (List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                Wavelet decomposition coefficients

        Returns:

            img (torch.Tensor): reconstruted image.
        """
        # TODO: Add special case for wavelet `self` 
        return ptwt.waverec2(coeffs, wavelet=self.wav).squeeze(1)


class DictionaryWv_torch(torch.nn.Module):
    """
    Constructs class to permit sparsity averaging across a collection of wavelet dictionaries
    """

    def __init__(self, wavs, levels, mode="periodic"):
        """Initialises a linear operator for a collection of abstract wavelet dictionaries

        Args:

            wavs (list[string]): List of wavelet types (see https://tinyurl.com/5n7wzpmb)
            levels (list[int]): Wavelet levels (scales) to consider
            mode (str): Wavelet signal extension mode shared by all dictionaries

        Raises:

            ValueError: Raised when levels are not positive definite

        """
        super().__init__()
        self.wavelet_list = []
        self.mode = mode
        self.wavs = wavs
        self.levels = levels
        if np.isscalar(levels):
            self.levels = np.ones(len(self.wavs)) * levels
        for i in range(len(self.wavs)):
            self.wavelet_list.append(
                Wavelets_torch(self.wavs[i], self.levels[i], self.mode)
            )

    def dir_op(self, x):
        """Evaluates a list of forward abstract wavelet transforms of x

        Args:

            x (torch.Tensor): Tensor to wavelet transform

        """
        buff = []
        buff.append(self.wavelet_list[0].dir_op(x))
        for wav_i in range(1, len(self.wavelet_list)):
            buff.append(self.wavelet_list[wav_i].dir_op(x))
        return buff

    def adj_op(self, coeffs):
        """Evaluates a list of forward adjoint abstract wavelet transforms of x

        Args:

            coeffs (List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                Coefficients to adjoint wavelet transform

        """
        out = self.wavelet_list[0].adj_op(coeffs[0])
        for wav_i in range(1, len(self.wavelet_list)):
            out = out + self.wavelet_list[wav_i].adj_op(coeffs[wav_i])
        return out / len(self.wavelet_list)


# TODO consider the special case of `self` wavelets
class L1Norm_torch(torch.nn.Module):
    """This class computes the proximity operator of the l2 ball.

                        f(x) = ||Psi x||_1 * gamma

    When the input 'x' is an array. gamma is a regularization term. Psi is a sparsity operator.
    """

    def __init__(self, gamma, Psi=None, op_to_coeffs=False):
        """Initialises an l1-norm proximal operator class

        Args:

            gamma (double >= 0): Regularisation parameter
            Psi (Linear operator): Regularisation functional (typically wavelets)

        Raises:

            ValueError: Raised if regularisation parameter is not postitive semi-definite
        """
        super().__init__()
        if np.any(gamma <= 0):
            raise ValueError("'gamma' must be positive semi-definite")

        self.gamma = gamma
        self.beta = 1.0
        self.op_to_coeffs = op_to_coeffs

        if Psi is None:
            self.Psi = Identity()
            self.num_wavs = 0
        else:
            self.Psi = Psi
            # Set the number of wavelets dictionaries
            self.num_wavs = len(self.Psi.wavelet_list)

        if self.op_to_coeffs:
            self.prox = self._prox_coeffs
            self.fun = self._fun_coeffs
        else:
            self.prox = self._prox
            self.fun = self._fun

    def _apply_op_to_coeffs(self, coeffs, op):
        """Applies operation to all coefficients in ptwt structure."""
        # Iterate over the wavelet dictionaries
        for wav_i in range(self.num_wavs):
            # Apply op over the low freq approx
            coeffs[wav_i][0] = op(coeffs[wav_i][0])
            # Iterate over the wavelet decomp and apply op
            for it1 in range(1, len(coeffs[0])):
                coeffs[wav_i][it1] = tuple([op(elem) for elem in coeffs[wav_i][it1]])

        return coeffs

    def _op_to_two_coeffs(self, coeffs1, coeffs2, op):
        """Applies operation to two set of coefficients in ptwt structure.

        Saves result in coeffs1.

        Args:

            coeffs1 (List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                First set of wavelet coefficients
            coeffs2 (List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                Second set of wavelet coefficients
            op (function): Operation to apply

        Returns:

            coeffs (List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                Resulting coefficients
        """
        # Iterate over the wavelet dictionaries
        for wav_i in range(self.num_wavs):
            # Apply op over the low freq approx
            coeffs1[wav_i][0] = op(coeffs1[wav_i][0], coeffs2[wav_i][0])
            # Iterate over the wavelet decomp and apply op
            for it1 in range(1, len(coeffs1[0])):
                coeffs1[wav_i][it1] = tuple(
                    [
                        op(elem1, elem2)
                        for elem1, elem2 in zip(
                            coeffs1[wav_i][it1], coeffs2[wav_i][it1]
                        )
                    ]
                )
        return coeffs1

    def _get_max_abs_coeffs(self, coeffs):
        """Get the max abs value of all coefficients

        Args:

            coeffs (List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                Set of wavelet coefficients

        Returns:

            max abs value of all coefficients
        """
        max_val = []
        # Iterate over the wavelet dictionaries
        for wav_i in range(self.num_wavs):
            # Apply op over the low freq approx
            max_val.append(torch.max(torch.abs((coeffs[wav_i][0]))))
            # Iterate over the wavelet decompositions
            for it1 in range(1, len(coeffs[0])):
                for it2 in range(len(coeffs[wav_i][it1])):
                    max_val.append(torch.max(torch.abs((coeffs[wav_i][it1][it2]))))

        # Apply operation to the coefficients
        return torch.max(torch.tensor(max_val)).item()

    def _prox_coeffs(self, x, tau):
        """Evaluates the l1-norm prox of Psi x

        Args:

            x (ptwt.coeffs): Array to evaluate proximal projection of
            tau (double): Custom weighting of l1-norm prox

        Returns:

            l1-norm prox of x
        """
        # Define the element-wise operation
        # op = partial(self._prox, tau=tau)
        op = lambda _x: self._prox(_x, tau=tau)
        # Apply operation to the coefficients
        return self._apply_op_to_coeffs(x, op)

    def _fun_coeffs(self, coeffs):
        """Evaluates loss of l1-norm regularisation for coeffs

        Args:

            x (ptwt.coeffs): Array to evaluate proximal projection of
            tau (double): Custom weighting of l1-norm prox

        Returns:

            l1-norm prox of x
        """
        loss = 0
        # Iterate over the wavelet dictionaries
        for wav_i in range(self.num_wavs):
            # Apply op over the low freq approx
            loss += self._fun(coeffs[wav_i][0])
            # Iterate over the wavelet decompositions
            for it1 in range(1, len(coeffs[0])):
                for it2 in range(len(coeffs[wav_i][it1])):
                    loss += self._fun(coeffs[wav_i][it1][it2])

        # Apply operation to the coefficients
        return loss

    def _prox(self, x, tau):
        """Evaluates the l1-norm prox of x.

        Args:

            x (torch.Tensor): Array to evaluate proximal projection of
            tau (double): Custom weighting of l1-norm prox

        Returns:

            l1-norm prox of x
        """
        # Replaced the use of torch.sign() to add complex value support
        abs_x = torch.abs(x)
        return torch.maximum(
            torch.zeros_like(abs_x), abs_x - self.gamma * tau
        ) * torch.nan_to_num(x / abs_x, nan=0.0)

        # return torch.maximum(
        #         torch.zeros_like(x), torch.abs(x) - self.gamma * tau
        #     ) * torch.exp(
        #         torch.complex(torch.tensor(0.), torch.tensor(1.)) * torch.angle(x)
        #     )

    def _fun(self, x):
        """Evaluates loss of functional term of l1-norm regularisation

        Args:

            x (torch.Tensor): Tensor to evaluate loss of

        Returns:

            l1-norm loss
        """
        return torch.sum(torch.abs(self.gamma * x))

    def dir_op(self, x):
        """Evaluates the forward regularisation operator

        Args:

            x (torch.Tensor): Tensor to forward transform

        Returns:

            Forward regularisation operator applied to x
        """
        return self.Psi.dir_op(x)

    def adj_op(self, x):
        """Evaluates the forward adjoint regularisation operator

        Args:

            x (torch.Tensor): Tensor to adjoint transform

        Returns:

            Forward adjoint regularisation operator applied to x
        """
        return self.Psi.adj_op(x)


class RealProx_torch(torch.nn.Module):
    """This class computes the proximity operator of the indicator function for
    reality.

                        f(x) = (Re{x} == x) ? 0. : infty
    it returns the projection.
    """

    def __init__(self):
        """
        Initialises a real half-plane proximal operator class
        """
        super().__init__()
        self.beta = 1.0

    def prox(self, x, tau=1.0):
        """Evaluates the real half-plane projection of x

        Args:

            x (torch.tensor): Array to evaluate proximal projection of

        Returns:

            real half-plane projection of x
        """
        return torch.real(x)

    def fun(self, x):
        """Evaluates loss of functional term

        Args:

            x (torch.tensor): Array to evaluate loss of

        Returns:

            0
        """
        return 0.0

    def dir_op(self, x):
        """Evaluates the forward operator

        Args:

            x (torch.tensor): Array to forward transform

        Returns:

            Forward operator applied to x
        """
        return x

    def adj_op(self, x):
        """Evaluates the forward adjoint operator

        Args:

            x (torch.tensor): Array to forward adjoint transform

        Returns:

            Forward adjoint operator applied to x
        """
        return x


class Operation2WaveletCoeffs_torch(torch.nn.Module):
    """This class helps to apply operations to wavelet coefficients."""

    def __init__(self, Psi=None):
        """Initialise

        Args:
            Psi (Linear operator): Wavelet transform class


        """
        super().__init__()

        if Psi is None:
            self.Psi = Identity()
            self.levels = 0
            self.num_wavs = 0
        else:
            self.Psi = Psi
            if type(self.Psi) is Wavelets_torch:
                # Number of wavelets used in the dictionary
                self.num_wavs = 1

            elif type(self.Psi) is DictionaryWv_torch:
                # Number of wavelets used in the dictionary
                self.num_wavs = len(self.Psi.wavelet_list)

            # Set the number of wavelets scales
            self.levels = self.Psi.levels

    def _apply_op_to_coeffs(self, coeffs, op):
        """Applies operation to all coefficients in ptwt structure."""
        # Iterate over the wavelet dictionaries
        for wav_i in range(self.num_wavs):
            # Apply op over the low freq approx
            coeffs[wav_i][0] = op(coeffs[wav_i][0])
            # Iterate over the wavelet decomp and apply op
            for it1 in range(1, len(coeffs[0])):
                coeffs[wav_i][it1] = tuple([op(elem) for elem in coeffs[wav_i][it1]])

        return coeffs

    def _apply_op_to_coeffs_at_level(self, coeffs, level, op):
        """Applies operation to all coefficients at a given level in ptwt structure.

        level (int or None): Level of wavelet decomposition to apply the operation.
            If the level is None, the operation is applied to all existing levels.
        """
        if level is None:
            coeffs = self._apply_op_to_coeffs(coeffs, op)
        else:
            # Iterate over the wavelet dictionaries
            for wav_i in range(self.num_wavs):
                if level == 0:
                    # Apply op over the low freq approx
                    coeffs[wav_i][0] = op(coeffs[wav_i][0])
                elif level > 0 and level <= len(coeffs[0]):
                    # Apply op to specific level
                    coeffs[wav_i][level] = tuple(
                        [op(elem) for elem in coeffs[wav_i][level]]
                    )
                else:
                    raise ValueError(
                        "The level requested is higher than the one used in the wavelet decomposition."
                    )

        return coeffs

    def _op_to_two_coeffs(self, coeffs1, coeffs2, op):
        """Applies operation to two set of coefficients in ptwt structure.

        Saves result in coeffs1.

        Args:

            coeffs1 (List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                First set of wavelet coefficients
            coeffs2 (List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                Second set of wavelet coefficients
            op (function): Operation to apply

        Returns:

            coeffs (List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                Resulting coefficients
        """
        # Iterate over the wavelet dictionaries
        for wav_i in range(self.num_wavs):
            # Apply op over the low freq approx
            coeffs1[wav_i][0] = op(coeffs1[wav_i][0], coeffs2[wav_i][0])
            # Iterate over the wavelet decomp and apply op
            for it1 in range(1, len(coeffs1[0])):
                coeffs1[wav_i][it1] = tuple(
                    [
                        op(elem1, elem2)
                        for elem1, elem2 in zip(
                            coeffs1[wav_i][it1], coeffs2[wav_i][it1]
                        )
                    ]
                )
        return coeffs1

    def _op_to_two_coeffs_at_level(self, coeffs1, coeffs2, level, op):
        """Applies operation to two set of coefficients in ptwt structure at a certain level.

        Saves result in coeffs1.

        Args:

            coeffs1 (List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                First set of wavelet coefficients
            coeffs2 (List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                Second set of wavelet coefficients
            level (int or None): Level of wavelet decomposition to apply the operation.
                If the level is None, the operation is applied to all existing levels.
            op (function): Operation to apply

        Returns:

            coeffs (List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                Resulting coefficients
        """
        if level is None:
            coeffs1 = self._op_to_two_coeffs(coeffs1, coeffs2, op)
        else:
            # Iterate over the wavelet dictionaries
            for wav_i in range(self.num_wavs):
                if level == 0:
                    # Apply op over the low freq approx
                    coeffs1[wav_i][0] = op(coeffs1[wav_i][0], coeffs2[wav_i][0])
                elif level > 0 and level <= len(coeffs1[0]):
                    # Apply op to specific level
                    coeffs1[wav_i][level] = tuple(
                        [
                            op(elem1, elem2)
                            for elem1, elem2 in zip(
                                coeffs1[wav_i][level], coeffs2[wav_i][level]
                            )
                        ]
                    )

        return coeffs1

    def _get_max_abs_coeffs(self, coeffs):
        """Get the max abs value of all coefficients

        Args:

            coeffs (List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                Set of wavelet coefficients

        Returns:

            max abs value of all coefficients
        """
        max_val = []
        # Iterate over the wavelet dictionaries
        for wav_i in range(self.num_wavs):
            # Apply op over the low freq approx
            max_val.append(torch.max(torch.abs((coeffs[wav_i][0]))))
            # Iterate over the wavelet decompositions
            for it1 in range(1, len(coeffs[0])):
                for it2 in range(len(coeffs[wav_i][it1])):
                    max_val.append(torch.max(torch.abs((coeffs[wav_i][it1][it2]))))

        # Apply operation to the coefficients
        return torch.max(torch.tensor(max_val)).item()

    def threshold_coeffs(self, coeffs, thresh, thresh_type, level=None):
        """Threshold coefficients and put to zero

        Args:

            coeffs (ptwt.coeffs): Wavelet coefficients
            thresh (double): Threshold
            thresh_type (str): type of thresholding. Options are: 'soft' or 'hard'.
            level (int or None): Level of wavelet decomposition to apply the operation.
                If the level is None, the operation is applied to all existing levels.

        Returns:

            Thresholded coefficients (ptwt.coeffs)
        """
        # Define the element-wise operation
        if thresh_type == "soft":
            op = lambda _x: self._threshold_soft(_x, thresh=thresh)
        elif thresh_type == "hard":
            op = lambda _x: self._threshold_hard(_x, thresh=thresh)
        # Apply operation to the coefficients
        return self._apply_op_to_coeffs_at_level(coeffs, level, op)

    def full_op_two_img(self, img1, img2, op, level=None):
        """Apply op to two image wavelet coefficients

        Args:

            img1 (torch.Tensor): Image n1 [H,W]
            img2 (torch.Tensor): Image n2 [H,W]
            op (function): Operation to apply
            level (int or None): Level of wavelet decomposition to apply the operation.
                If the level is None, the operation is applied to all existing levels.

        Returns:

            Modified img (torch.Tensor)
        """
        return self.adj_op(
            self._op_to_two_coeffs_at_level(
                self.dir_op(img1), self.dir_op(img2), level=level, op=op
            )
        ).squeeze()

    def full_op_threshold_img(self, img, thresh, level=None, thresh_type="soft"):
        """Threshold image wavelet coefficients

        Args:

            img (torch.Tensor): Image [H,W]
            thresh (double): Threshold
            level (int or None): Level of wavelet decomposition to apply the operation.
                If the level is None, the operation is applied to all existing levels.
            thresh_type (str): type of thresholding. Options are: 'soft' or 'hard'.

        Returns:

            Thresholded img (torch.Tensor)
        """
        return self.adj_op(
            self.threshold_coeffs(
                self.dir_op(img), thresh=thresh, thresh_type=thresh_type, level=level
            )
        ).squeeze()

    def full_op_add_img(self, img, val, level=None):
        """Add val to image wavelet coefficients at given level

        Args:

            img (torch.Tensor): Image [H,W]
            thresh (double): Threshold
            level (int or None): Level of wavelet decomposition to apply the operation.
                If the level is None, the operation is applied to all existing levels.

        Returns:

            Modified img (torch.Tensor)
        """
        return self.adj_op(
            self.add_value_at_level(self.dir_op(img), level=level, val=val)
        ).squeeze()

    def full_op_mult_img(self, img, val, level=None):
        """Multiply val to image wavelet coefficients at given level

        Args:

            img (torch.Tensor): Image [H,W]
            thresh (double): Threshold
            level (int or None): Level of wavelet decomposition to apply the operation.
                If the level is None, the operation is applied to all existing levels.

        Returns:

            Modified img (torch.Tensor)
        """
        return self.adj_op(
            self.mult_value_at_level(self.dir_op(img), level=level, val=val)
        ).squeeze()

    def add_value_at_level(self, coeffs, level, val):
        """Threshold coefficients and put to zero

        Args:

            coeffs (ptwt.coeffs): Wavelet coefficients
            level (int or None): Level of wavelet decomposition to apply the operation.
                If the level is None, the operation is applied to all existing levels.
            val (double): value

        Returns:

            Modified coefficients (ptwt.coeffs)
        """
        # Define the element-wise operation
        op = lambda _x: _x + val
        # Apply operation to the coefficients
        return self._apply_op_to_coeffs_at_level(coeffs, level, op)

    def mult_value_at_level(self, coeffs, level, val):
        """Threshold coefficients and put to zero

        Args:

            coeffs (ptwt.coeffs): Wavelet coefficients
            level (int or None): Level of wavelet decomposition to apply the operation.
                If the level is None, the operation is applied to all existing levels.
            val (double): value

        Returns:

            Modified coefficients (ptwt.coeffs)
        """
        # Define the element-wise operation
        op = lambda _x: _x * val
        # Apply operation to the coefficients
        return self._apply_op_to_coeffs_at_level(coeffs, level, op)

    def _threshold_soft(self, x, thresh):
        """Threshold coefficients (soft)

        Args:

            x (torch.Tensor): tensor to operate on
            thresh (double): Threhsold

        Returns:

            Thresholded version of x
        """
        # Replaced the use of torch.sign() to add complex value support
        abs_x = torch.abs(x)
        return torch.maximum(
            torch.zeros_like(abs_x), abs_x - thresh
        ) * torch.nan_to_num(x / abs_x, nan=0.0)

    def _threshold_hard(self, x, thresh):
        """Threshold coefficients (hard)

        Args:

            x (torch.Tensor): tensor to operate on
            thresh (double): Threhsold

        Returns:

            Thresholded version of x
        """
        out = x.clone()
        out[abs(out) < thresh] = 0
        return out

    def dir_op(self, x):
        """Evaluates the forward regularisation operator

        Args:

            x (torch.Tensor): Tensor to forward transform

        Returns:

            Forward regularisation operator applied to x
        """
        return self.Psi.dir_op(x)

    def adj_op(self, x):
        """Evaluates the forward adjoint regularisation operator

        Args:

            x (torch.Tensor): Tensor to adjoint transform

        Returns:

            Forward adjoint regularisation operator applied to x
        """
        return self.Psi.adj_op(x)
