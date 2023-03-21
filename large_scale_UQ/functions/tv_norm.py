
#    Computing the TV norm.
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

def tv(Dx):
    
    with torch.no_grad():

        Dx=Dx.view(-1)
        N = len(Dx)
        Dux = Dx[:int(N/2)]
        Dvx = Dx[int(N/2):N]
        tv = torch.sum(torch.sqrt(Dux**2 + Dvx**2))
        
        return tv
