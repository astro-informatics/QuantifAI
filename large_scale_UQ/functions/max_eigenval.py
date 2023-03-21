# Computes the maximum eigen value of the compund 
# operator AtA

# Written by : A. F. Vidal

# GitHub account https://github.com/anafvidal/research-code

# [1] A. F. Vidal, V. De Bortoli, M. Pereyra, and A. Durmus (2020). 
# Maximum Likelihood Estimation of Regularization Parameters in High-Dimensional Inverse Problems: 
# An Empirical Bayesian Approach Part I: Methodology and Experiments. 
# SIAM Journal on Imaging Sciences, 13(4), 1945-1989.

#    Adapted in pytorch by: MI2G
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

import numpy as np
from scipy.linalg import norm
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def max_eigenval(A, At, im_size, tol, max_iter, verbose):

    with torch.no_grad():

        #computes the maximum eigen value of the compund operator AtA
        
        x = torch.normal(mean=0, std=1,size=(im_size,im_size))[None][None].to(device)
        x = x/torch.norm(torch.ravel(x),2)
        init_val = 1
        
        for k in range(0,max_iter):
            y = A(x)
            x = At(y)
            val = torch.norm(torch.ravel(x),2)
            rel_var = torch.abs(val-init_val)/init_val
            if (verbose > 1):
                print('Iter = {}, norm = {}',k,val)
            
            if (rel_var < tol):
                break
            
            init_val = val
            x = x/val
        
        if (verbose > 0):
            print('Norm = {}', val)
        
        return val