
#    Compute the gradient operator of an image
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

def Grad_Image(x):

    with torch.no_grad():

        x = x.to(device).clone()
        x_temp = x[1:, :] - x[0:-1,:]
        dux = torch.cat((x_temp.T,torch.zeros(x_temp.shape[1],1,device=device)),1).to(device)
        dux = dux.T
        x_temp = x[:,1:] - x[:,0:-1]
        duy = torch.cat((x_temp,torch.zeros((x_temp.shape[0],1),device=device)),1).to(device)
        return  torch.cat((dux,duy),dim=0).to(device)
