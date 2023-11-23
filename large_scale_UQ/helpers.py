
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

def get_hypothesis_test_mask(img_name, physical=True):
    """Load hypothesis test masks.

    Follows X. Cai article's areas.

    Args:
        img_name (str): Radio image name
        physical (bool): if the area contains phyisical information

    Returns:
        mask_x (list): mask's x coordinates
        mask_y (list): mask's y coordinates
    """
    if img_name == 'M31':
        if physical:
            mask_x = np.array([143, 225])
            mask_y = np.array([29, 200])
        else:
            raise NotImplementedError

    elif img_name == 'CYN':
        if physical:
            mask_x = np.array([124, 157])
            mask_y = np.array([219, 256])
        else:
            raise NotImplementedError

    elif img_name == 'W28':
        if physical:
            mask_x = np.array([87, 119])
            mask_y = np.array([9, 39])
        else:
            raise NotImplementedError

    elif img_name == '3c288':
        if physical:
            mask_x = np.array([118, 140])
            mask_y = np.array([87, 119])
        else:
            mask_x = np.array([14, 34])
            mask_y = np.array([156, 180])           

    return mask_x, mask_y

def compute_complex_sigma_noise(observations, input_snr):
    """Computes the standard deviation of a complex Gaussian noise

    The eff_sigma is such that `Im(n), Re(n) \sim N(0,eff_sigma)`, where
    eff_sigma=sigma/sqrt(2)

    Args:
        observations (np.ndarray): complex observations
        input_snr (float): desired input SNR

    Returns:
        eff_sigma (float): effective standard deviation for the complex Gaussian noise
    """
    num_measurements = observations[observations!=0].shape[0]

    sigma = 10**(-input_snr/20)*(
        np.linalg.norm(observations.flatten(),ord=2)/np.sqrt(num_measurements)
    )
    eff_sigma = sigma / np.sqrt(2)

    return eff_sigma

