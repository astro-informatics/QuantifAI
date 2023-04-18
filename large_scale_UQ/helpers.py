
import numpy as np
import scipy.io as sio
from astropy.io import fits
from skimage import transform


def load_imgs(img_name, repo_dir):
    """Load radio images.
    
    Load radio image and the corresponding Fourier mask.
    The preprocessing follows X. Cai's implementation used for his article.

    Args:
        img_name (str): Radio image name
        repo_dir (str): path to the repository directory

    Returns:
        img (np.ndarray): preprocessed radio image
        mat_mask (np.ndarray): Fourier mask
    """

    # Load img
    img_path = repo_dir + '/data/imgs/{:s}.fits'.format(img_name)
    img_data = fits.open(img_path, memmap=False)

    # Loading the image and cast it to float
    img = np.squeeze(np.copy(img_data[0].data)).astype(np.float64)
    # Flipping data
    img = np.flipud(img)

    # Resize image 
    if img_name == 'CYN':
        img = transform.resize(img, [256, 512], order=3, anti_aliasing=True)

    if img_name == 'W28' or img_name == '3c288':
        img = transform.resize(img, [256, 256], order=3, anti_aliasing=True)

    # Normalise
    img = img - np.min(img)
    img = img / np.max(img)

    # Load op from X Cai
    op_mask = sio.loadmat(
        repo_dir + '/data/operators_masks/' + img_name + '_fourier_mask.mat'
    )['Ma']

    # Matlab's reshape works with 'F'-like ordering
    mat_mask = np.reshape(np.sum(op_mask, axis=0), img.shape, order='F').astype(bool)

    return img, mat_mask
