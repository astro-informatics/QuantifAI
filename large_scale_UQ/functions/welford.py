#    Welford's algorithm for calculating mean and variance
#    Collection of summary statistics of the Markov chain iterates
# 
#    call update() when generating a new sample
#    call get_mean() or get_var() to obtain the current mean or variance
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
#
# https://doi.org/10.2307/1266577

class welford: 
    def __init__(self, x):
        self.k = 1
        self.M = x.clone()
        self.S = 0

    def update(self, x):
        self.k += 1
        Mnext = self.M + (x - self.M) / self.k
        self.S += (x - self.M)*(x - Mnext)
        self.M = Mnext
    
    def get_mean(self):
        return self.M
    
    def get_var(self):
        #when the number of samples is 1, we divide by self.k 
        if self.k < 2:
            return self.S/(self.k)
        else:
            return self.S/(self.k-1)
