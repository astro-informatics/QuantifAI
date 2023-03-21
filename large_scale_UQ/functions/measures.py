#    Wrapper functions for computing the normalized root mean 
#    square error (NRMSE), structural similarity index (SSIM),
#    peak signal to noise ratio (PSNR).
#
#    Copyright (C) 2023 MI2G
#    Dobson, Paul pdobson@ed.ac.uk
#    Kemajou, Mbakam Charlesquin cmk2000@hw.ac.uk
#    Klatzer, Teresa t.klatzer@sms.ed.ac.uk
#    Melidonis, Savvas sm2041@hw.ac.uk
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def to_numpy(x):
    return x.detach().cpu().numpy().squeeze()

def NRMSE(x, y):
    x_np = to_numpy(x)
    return np.linalg.norm(x_np - to_numpy(y),'fro')/np.linalg.norm(x_np,'fro')

def SSIM(x, y):
    return ssim(to_numpy(x), to_numpy(y), data_range=1)

def PSNR(x, y):
    return psnr(to_numpy(x), to_numpy(y), data_range=1)

