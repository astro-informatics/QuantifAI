# 
import numpy as np
import math

def FindItv4Area(sol, areas, stepSize, energyG, gamma_alpha_min):
    # Given area
    xIdxL, xIdxH, yIdxL, yIdxH = areas
    
    # Compute the mean value of the given area
    temp = sol[xIdxL:xIdxH+1, yIdxL:yIdxH+1]
    meanVal = temp.mean()

    # Find a lower bound
    bound_l = meanVal
    energy_dummy = 0
    while energy_dummy <= gamma_alpha_min:
        bound_l = bound_l - stepSize

        if bound_l <= 0:
            bound_l = 0
            break

        dummy = sol.copy()
        dummy[xIdxL:xIdxH+1, yIdxL:yIdxH+1] = bound_l
        energy_dummy = energyG(dummy)

    # Find an upper bound
    bound_h = meanVal
    energy_dummy = 0
    while energy_dummy <= gamma_alpha_min:
        bound_h = bound_h + stepSize

        if bound_h >= 1e3:
            break

        dummy = sol.copy()
        dummy[xIdxL:xIdxH+1, yIdxL:yIdxH+1] = bound_h
        energy_dummy = energyG(dummy)

    return bound_l, bound_h


def find_itv4_grid_img(grid_size, sol, step_size, energy_g, gamma_alpha_min):
    """
    This program is to find an credible interval for an image according to
    given grid

    Args:
        gridSize (list[int]): grid used to do gridding on image
        sol (numpy.ndarray): optimal solution of given minimization problem
        stepSize (list[float]): step size used to search credible region
        energyG (function): operator to compute energy of given problem
        gamma_alpha_min (float): energy upper bound

    Returns:
        tuple(numpy.ndarray, numpy.ndarray): lower and upper bounds computed - way 1
    """
    x_len, y_len = sol.shape
    gridImg1_lh = []
    
    for j in range(len(grid_size)):
        gridImgTemp1_l = np.zeros((x_len, y_len))
        gridImgTemp1_h = np.zeros((x_len, y_len))
        gst = grid_size[j]
        x_grid_len = math.ceil(x_len / gst)
        y_grid_len = math.ceil(y_len / gst)

        for jj in range(1, x_grid_len + 1):
            for kk in range(1, y_grid_len + 1):
                areas = {}
                areas['xIdxL'] = (jj-1) * gst + 1
                areas['xIdxH'] = jj * gst
                areas['yIdxL'] = (kk-1) * gst + 1
                areas['yIdxH'] = kk * gst

                if areas['xIdxH'] > x_len:
                    areas['xIdxH'] = x_len
                    areas['xIdxL'] = x_len - gst + 1

                if areas['yIdxH'] > y_len:
                    areas['yIdxH'] = y_len
                    areas['yIdxL'] = y_len - gst + 1

                cr_bound_l, cr_bound_h = FindItv4Area(sol, areas, step_size[j], energy_g, gamma_alpha_min)
                gridImgTemp1_l[areas['xIdxL']:areas['xIdxH'], areas['yIdxL']:areas['yIdxH']] = cr_bound_l
                gridImgTemp1_h[areas['xIdxL']:areas['xIdxH'], areas['yIdxL']:areas['yIdxH']] = cr_bound_h

        gridImg1_lh.append((gridImgTemp1_l, gridImgTemp1_h))


    return gridImg1_lh



def FindItv4GridImg(gridSize, sol, stepSize, energyG, gamma_alpha_min):
    """
    This program is to find an credible interval for an image according to given grid.

    Usage: [gridImg1_lh, gridImg2_lh] = FindItv4GridImg(gridSize, sol, stepSize, energyG, gamma_alpha_min)

    Inputs:
        - gridSize: grid used to do griding on image
        - sol: optimal solution of given minimisation problem
        - stepSize: step size used to search credible region
        - energyG: operator to compute energy of given problem
        - gamma_alpha_min: energy upper bound

    Outputs:
        - gridImg1_lh: lower and upper bounds computed - way 1
        - gridImg2_lh: lower and upper bounds computed - way 2
    """

    xLen, yLen = sol.shape

    for j in range(len(gridSize)):
        gridImgTemp1_l = np.zeros((xLen, yLen))
        gridImgTemp1_h = np.zeros((xLen, yLen))
        gst = gridSize[j]
        xGridLen = int(np.ceil(xLen / gst))
        yGridLen = int(np.ceil(yLen / gst))

        for jj in range(1, xGridLen + 1):
            for kk in range(1, yGridLen + 1):
                areas = {}
                areas['xIdxL'] = (jj-1) * gst + 1
                areas['xIdxH'] = jj * gst
                areas['yIdxL'] = (kk-1) * gst + 1
                areas['yIdxH'] = kk * gst

                if areas['xIdxH'] > xLen:
                    areas['xIdxH'] = xLen
                    areas['xIdxL'] = xLen - gst + 1

                if areas['yIdxH'] > yLen:
                    areas['yIdxH'] = yLen
                    areas['yIdxL'] = yLen - gst + 1

                cr_bound_l, cr_bound_h = FindItv4Area(sol, areas, stepSize[j], energyG, gamma_alpha_min)
                gridImgTemp1_l[areas['xIdxL']:areas['xIdxH'], areas['yIdxL']:areas['yIdxH']] = cr_bound_l
                gridImgTemp1_h[areas['xIdxL']:areas['xIdxH'], areas['yIdxL']:areas['yIdxH']] = cr_bound_h

        gridImg1_lh = {'cr_bound_l': gridImgTemp1_l, 'cr_bound_h': gridImgTemp1_h}


    return gridImg1_lh
