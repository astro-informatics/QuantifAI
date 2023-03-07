

function [gridImg1_lh, gridImg2_lh] = FindItv4GridImg(gridSize, sol, ...
    stepSize, energyG, gamma_alpha_min)
%--------------------------------------------------------------------------
%   This program is to find an credible interval for an image accrding to
%        given gird 
%    
%   Usage: gridImg_lh = FindItv4GridImg(gridSize, sol, stepSize, ...
%        energyG, gamma_alpha_min)
%
%   Inputs: 
%       - gridSize: grid used to do griding on image
%       - sol_ana: optimal solution of given minimisation problem
%       - stepSize: step size used to search credible region
%       - energyG: operator to compute energy of given problem
%       - gamma_alpha_min: energy upper bound 
%
%   Outputs: 
%       - gridImg1_lh: lower and upper bounds computed - way 1
%       - gridImg2_lh: lower and upper bounds computed - way 2
% 
%--------------------------------------------------------------------------

[xLen,yLen] = size(sol);

for j=1:length(gridSize)
    gridImgTemp1_l = zeros(xLen,yLen);
    gridImgTemp1_h = zeros(xLen,yLen);
    gridImgTemp2_l = zeros(xLen,yLen);
    gridImgTemp2_h = zeros(xLen,yLen);
    gst = gridSize(j);
    xGridLen = ceil(xLen/gst);
    yGridLen = ceil(yLen/gst);
    
    for jj=1:xGridLen
        for kk=1:yGridLen       
            areas.xIdxL = (jj-1)*gst+1; areas.xIdxH = jj*gst;
            areas.yIdxL = (kk-1)*gst+1; areas.yIdxH = kk*gst;
            
            if areas.xIdxH > xLen
                areas.xIdxH = xLen;
                areas.xIdxL = xLen-gst+1;              
            end

            if areas.yIdxH > yLen
                areas.yIdxH = yLen;
                areas.yIdxL = yLen-gst+1;
            end

            [cr_bound_l, cr_bound_h] = FindItv4Area(sol, areas, ...
                stepSize(j), energyG, gamma_alpha_min);
            gridImgTemp1_l(areas.xIdxL:areas.xIdxH, areas.yIdxL:areas.yIdxH) = cr_bound_l;
            gridImgTemp1_h(areas.xIdxL:areas.xIdxH, areas.yIdxL:areas.yIdxH) = cr_bound_h;  

        end
    end
    
    gridImg1_lh{j}.cr_bound_l = gridImgTemp1_l;
    gridImg1_lh{j}.cr_bound_h = gridImgTemp1_h;

end

gridImg2_lh = 0;
