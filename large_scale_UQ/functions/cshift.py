
#    Implementing circular shift in torch. Used for blur operators.
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

import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cshift(x,L):

    with torch.no_grad():

        N = len(x)
        y = torch.zeros(N)
        
        if L == 0:
            y = x.clone().detach()
            return y
        
        if L > 0:
            y[L:] = x[0:N-L]
            y[0:L] = x[N-L:N]
        else:
            L=int(-L)
            y[0:N-L] = x[L:N]
            y[N-L:N] = x[0:L]
            
        return y           
