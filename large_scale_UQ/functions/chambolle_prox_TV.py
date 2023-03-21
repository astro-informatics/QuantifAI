# Proximal  point operator for the TV regularizer 

# Uses the Chambolle's projection  algorithm proposed in:

# "An Algorithm for Total Variation Minimization and
# Applications", J. Math. Imaging Vis., vol. 20, pp. 89-97, 2004.

#  Optimization problem:  

#     arg min = (1/2) || y - x ||_2^2 + lambda TV(x)
#         x

#   =========== Required inputs ====================

#  'g'       : noisy image (size X: ny * nx)

#   =========== Optional inputs ====================
  
#  'lambda'  : regularization  parameter according

#  'maxiter' :maximum number of iterations
  
#  'tol'     : tol for the stopping criterion

#  'tau'     : algorithm parameter

#  'dualvars' : dual variables: used to start the algorithm closer
#               to the solution. 
#               Input format: [px, py] where px amd py have the same size 
#               of g
            
  
#  =========== Outputs ====================

#  g - lambd * DivergenceIm(px,py)  : denoised image

#  ===================================================

#  Adapted by: Jose Bioucas-Dias, June 2009, (email: bioucas@lx.it.pt)
#  from Chambolle_Exact_TV(g, varargin) written by  Dr.Wen Youwei, email: wenyouwei@graduate.hku.hk

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

# Cuda 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def GradientIm(u):
    u_shapex = list(u.shape)
    u_shapex[0] = 1
    z = u[1:,:] - u[:-1,:]
    dux = torch.vstack([z, torch.zeros(u_shapex, device = device)])

    u_shapey = list(u.shape)
    u_shapey[1] = 1
    z = u[:,1:] - u[:,:-1]
    duy = torch.hstack([z, torch.zeros(u_shapey, device = device)])
    return dux, duy

def DivergenceIm(p1, p2):
    z = p2[:,1:-1] - p2[:,:-2]
    shape2 = list(p2.shape)
    shape2[1]=1
    v = torch.hstack([p2[:,0].reshape(shape2), z, -p2[:,-1].reshape(shape2)])
    
    shape1 = list(p1.shape)
    shape1[0]=1
    z = p1[1:-1,:] - p1[:-2,:]
    u = torch.vstack([p1[0,:].reshape(shape1), z, -p1[-1,:].reshape(shape1)])

    return v+u



def chambolle_prox_TV(g1, varargin):
  with torch.no_grad():

    g = g1.clone().detach()

    # initialize
    px = torch.zeros(g.shape, device = device)
    py = torch.zeros(g.shape, device = device)
    cont = 1     
    k    = 0

    #defaults for optional parameters
    tau = 0.249
    tol = 1e-3
    lambd = 1
    maxiter = 10
    verbose = 0
   
    #read the optional parameters
    for key in varargin.keys():
        if key.upper() == 'LAMBDA':
            lambd = varargin[key]
        elif key.upper() == 'VERBOSE':
            verbose = varargin[key]
        elif key.upper() == 'TOL':
            tol = varargin[key]
        elif key.upper() == 'MAXITER':
            maxiter = varargin[key]
        elif key.upper() == 'TAU':
            tau = varargin[key]
        elif key.upper() == 'DUALVARS':
            M,N = g.shape
            Maux, Naux = varargin[key].shape
            if M != Maux or N != 2*Naux:
                print('Wrong size of the dual variables')
                return
            px = torch.tensor(varargin[key])
            py = px[:,M:]
            px = px[:, 1:M]
        else:
            pass

    ## Main body
    while cont:
      k = k+1
      # compute Divergence of (px, py)
      divp = DivergenceIm(px,py) 
      u = divp - torch.divide(g, lambd).to(device)
      # compute gradient of u
      upx,upy = GradientIm(u)

      tmp = torch.sqrt(upx*upx + upy*upy).to(device)
      #error
      x1 = -upx.reshape(-1,1) + tmp.reshape(-1,1) * px.reshape(-1,1)
      y1 = -upy.reshape(-1,1) + tmp.reshape(-1,1) * py.reshape(-1,1)
      err = torch.sqrt(torch.sum(x1**2 + y1**2))

      # update px and py
      px = torch.divide(px + tau * upx,1 + tau * tmp).to(device)
      py = torch.divide(py + tau * upy,1 + tau * tmp).to(device)
      # check of the criterion
      cont = ((k<maxiter) and (err>tol))

    if verbose:
      print(f'\t\t|=====> k = {k}\n')
      print(f'\t\t|=====> err TV = {round(err,3)}\n')

    return g - lambd * DivergenceIm(px,py)
