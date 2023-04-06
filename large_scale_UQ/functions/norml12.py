# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:45:30 2021

@author: SavvasM
"""

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