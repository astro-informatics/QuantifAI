
import numpy as np
import torch
from large_scale_UQ.utils import max_eigenval


class Identity:
    """Identity operator

    Notes:
        Implemented originally in optimus-primal.
    """
    def __init__(self):
        """Initialises an identity operator class"""

    def dir_op(self, x):
        """Computes the forward operator of the identity class.

        Args:
            x (np.ndarray): Vector to apply identity to.

        Returns:
            np.ndarray: array of coefficients
        """
        return x

    def adj_op(self, x):
        """Computes the forward adjoint operator of the identity class.

        Args:
            x (np.ndarray): Vector to apply identity to.

        Returns:
            np.ndarray: array of coefficients
        """
        return x


class MaskedFourier:
    """
    Masked fourier sensing operator i.e. MRI/Radio imaging.
    """

    def __init__(
            self,
            dim,
            ratio=0.5,
            mask=None,
            norm=None,
            framework='numpy',
            device='cpu'
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
        if self.framework == 'pytorch':
            self.mask = torch.tensor(
                np.copy(mask), device=device
            ).reshape((1,1) + self.shape)

        if self.framework == 'numpy':
            self.dir_op = self._numpy_dir_op
            self.adj_op = self._numpy_adj_op
        elif self.framework == 'pytorch':
            self.dir_op = self._torch_dir_op
            self.adj_op = self._torch_adj_op


    def set_mask(self, new_mask):
        """Set new mask taking care of the framework."""
        if isinstance(new_mask, np.ndarray):
            if self.framework == 'numpy':
                self.mask = new_mask
            elif self.framework == 'pytorch':
                self.mask = torch.tensor(
                    np.copy(new_mask), device=self.device
                ).reshape((1,1) + new_mask.shape)
        elif isinstance(new_mask, torch.Tensor):
            if self.framework == 'numpy':
                self.mask = new_mask.detach().cpu().squeeze().numpy()
            elif self.framework == 'pytorch':
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

        return torch.mul(
            torch.fft.fft2(x, norm=self.norm),
            self.mask
        )
    
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

    def __init__(
            self,
            dim,
            ratio=0.5,
            mask=None,
            norm=None,
            device='cpu'
        ):
        """Initialises the masked fourier sensing operator.

        Args:
            dim (int): Dimension of square pixel-space image.
            ratio (float): Fraction of measurements observed.
            norm (str): FFT normalization mode. Options are `forward`, `backward` or `norm`.
            device (str): device for the `pytorch` framework.
        """
        super().__init__()
        self.norm = norm
        self.dim = dim
        self.ratio = ratio
        self.device = device
        self.shape = (dim, dim)
        self.mask = mask
        # If the mask is not defined, initialise a random one
        if mask is None:
            self.init_mask()
        else:
            # Check the channel dimensions for the pytorch framework
            if self.mask.shape != (1, 1, dim, dim):
                self.mask.reshape((1,1) + self.shape)
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
        mask = np.full(self.dim**2, False)
        mask[: int(self.ratio * self.dim**2)] = True
        np.random.shuffle(mask)
        self.mask = torch.tensor(
            np.copy(mask.reshape(self.shape)), device=self.device
        ).reshape((1,1) + self.shape)

    def set_mask(self, new_mask):
        """Set new mask."""
        if isinstance(new_mask, np.ndarray):
            self.mask = torch.tensor(
                np.copy(new_mask), device=self.device
            ).reshape((1,1) + self.shape)
        elif isinstance(new_mask, torch.Tensor):
                self.mask = new_mask.reshape((1,1) + self.shape)

    def _torch_dir_op(self, x):
        """Computes the forward operator of the class.
        
        Compute FFT and then mask Fourier coefficients.
        Using the `pytorch` framework.

        Args:
            x (torch.Tensor): Array to apply FFT and mask. Torch shape: (1, 1, dim, dim)

        Returns:
            torch.Tensor: tensor of Fourier coefficients. Same shape as input.
        """

        return torch.mul(
            torch.fft.fft2(x, norm=self.norm),
            self.mask
        )
    
    def _torch_adj_op(self, x):
        """Computes the forward adjoint operator of the class.

        Compute the inverse FFT.
        
        Args:
            x (torch.Tensor): tensor of Fourier coefficients. Same shape as output.

        Returns:
            torch.Tensor: Output array. Torch shape: (1, 1, dim, dim)
        """

        return torch.fft.ifft2(x, norm=self.norm)
    

class l2_norm_torch(torch.nn.Module):
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
            self.beta = 1. / sigma ** 2
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
        A = lambda _x : self.Phi.dir_op(_x)
        At = lambda _x : self.Phi.adj_op(_x)
        max_val = max_eigenval(
            A=A,
            At=At,
            im_size=self.data.shape[3],
            tol=1e-4,
            max_iter=int(1e4),
            verbose=0,
            device=self.data.device
        )
        # Store Lipschitz constant
        self.beta = max_val.item() / self.sigma ** 2


    def grad(self, x):
        """Computes the gradient of the l2_norm class

        Args:

            x (np.ndarray): Data estimate

        Returns:

            Gradient of the l2_norm expression

        """
        return self.Phi.adj_op((self.Phi.dir_op(x) - self.data)) / (
            self.sigma ** 2
        )

    def fun(self, x):
        """Evaluates the l2_norm class

        Args:

            x (torch.Tensor): Data estimate

        Returns:

            Computes the l2_norm loss

        """
        return torch.sum(torch.abs(self.data - self.Phi.dir_op(x)) ** 2.0) / (
            2 * self.sigma ** 2
        )        

