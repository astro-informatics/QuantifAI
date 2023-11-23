# Defining the blurring operators for the deblurring inverse problems as well as
# handle functions to calculate A*x in the Fourier domain.

# AUTHORS: Jizhou Li, Florian Luisier and Thierry Blu

# GitHub account : https://github.com/hijizhou/PureLetDeconv

# REFERENCES:
#     [1] J. Li, F. Luisier and T. Blu, PURE-LET image deconvolution,
#         IEEE Trans. Image Process., vol. 27, no. 1, pp. 92-105, 2018.
#     [2] J. Li, F. Luisier and T. Blu, Deconvolution of Poissonian images with the PURE-LET approach,
#         2016 23rd Proc. IEEE Int. Conf. on Image Processing (ICIP 2016), Phoenix, Arizona, USA, 2016, pp.2708-2712.
#     [3] J. Li, F. Luisier and T. Blu, PURE-LET deconvolution of 3D fluorescence microscopy images,
#         2017 14th Proc. IEEE Int. Symp. Biomed. Imaging (ISBI 2017), Melbourne, Australia, 2017, pp. 723-727.

# Adapted in pytorch by: MI2G

# Copyright (C) 2023 MI2G
# Dobson, Paul pdobson@ed.ac.uk
# Kemajou, Mbakam Charlesquin cmk2000@hw.ac.uk
# Klatzer, Teresa t.klatzer@sms.ed.ac.uk
# Melidonis, Savvas sm2041@hw.ac.uk
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import numpy as np
from large_scale_UQ.functions.max_eigenval import max_eigenval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def blur_operators(kernel_len, size, type_blur, var=None):
    nx = size[0]
    ny = size[1]
    if type_blur == "uniform":
        h = torch.zeros(nx, ny)
        lx = kernel_len[0]
        ly = kernel_len[1]
        h[0:lx, 0:ly] = 1 / (lx * ly)
        c = np.ceil((np.array([ly, lx]) - 1) / 2).astype("int64")
    if type_blur == "gaussian":
        if var != None:
            [x, y] = torch.meshgrid(
                torch.arange(-ny / 2, ny / 2), torch.arange(-nx / 2, nx / 2)
            )
            h = torch.exp(-(x**2 + y**2) / (2 * var))
            h = h / torch.sum(h)
            c = np.ceil(np.array([nx, ny]) / 2).astype("int64")
        else:
            print("Choose a variance for the Gaussian filter.")

    H_FFT = torch.fft.fft2(torch.roll(h, shifts=(-c[0], -c[1]), dims=(0, 1))).to(device)
    HC_FFT = torch.conj(H_FFT).to(device)

    # A forward operator
    A = lambda x: torch.fft.ifft2(
        torch.multiply(H_FFT, torch.fft.fft2(x))
    ).real.reshape(x.shape)

    # A backward operator
    AT = lambda x: torch.fft.ifft2(
        torch.multiply(HC_FFT, torch.fft.fft2(x))
    ).real.reshape(x.shape)

    AAT_norm = max_eigenval(A, AT, nx, 1e-4, int(1e4), 0)

    return A, AT, AAT_norm
