function [bound_l, bound_h] = FindItv4Area(sol,areas,stepSize, ...
    energyG, gamma_alpha_min)
%--------------------------------------------------------------------------
%   This program is to find an credible interval for a specified area or 
%       an indivial pixel
%    
%   Usage: [bound_b, bound_u] = FindItv4Area(sol,areas,stepSize, ...
%        energyG, gamma_alpha_min);
%
%   Inputs: 
%       - sol_ana: optimal solution of given minimisation problem
%       - areas: given area of interest, and find an credible region for it
%       - stepSize: step size used to search credible region
%       - energyG: operator to compute energy of given problem
%       - gamma_alpha_min: energy upper bound 
%
%   Outputs: 
%       - bound_l: lower bound computed
%       - bound_h: upper bound computed
% 
%--------------------------------------------------------------------------

dummy = sol;

% given area
xIdxL = areas.xIdxL; xIdxH = areas.xIdxH; 
yIdxL = areas.yIdxL; yIdxH = areas.yIdxH;

% compute the mean value of given area
temp = sol(xIdxL:xIdxH, yIdxL:yIdxH);
meanVal = mean(temp(:));

% find a lower bound
bound_l = meanVal;
energy_dummy = 0;
while energy_dummy <= gamma_alpha_min
    bound_l = bound_l - stepSize;

    if bound_l <=0
        bound_l = 0;
        break;
    end

    dummy(xIdxL:xIdxH, yIdxL:yIdxH) = bound_l;
    energy_dummy = energyG(dummy);    
end

% find an upper bound
bound_h = meanVal;
energy_dummy = 0;
while energy_dummy <= gamma_alpha_min
    bound_h = bound_h + stepSize;

    if bound_h >= 1e3
        break;
    end

    dummy(xIdxL:xIdxH, yIdxL:yIdxH) = bound_h;
    energy_dummy = energyG(dummy);    
end
