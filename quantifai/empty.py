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


class EmptyProx:
    def fun(self, x):
        return 0

    def prox(self, x, tau):
        return x

    def dir_op(self, x):
        return x

    def adj_op(self, x):
        return x

    beta = 1


class EmptyGrad:
    def fun(self, x):
        return 0

    def grad(self, x):
        return 0

    beta = 1
