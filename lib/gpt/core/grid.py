#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import cgpt
import gpt
import numpy as np

class full:
    n=1

class redblack:
    n=2

class grid:
    def __init__(self, gdimensions, precision, cb = full, obj = None):
        self.gdimensions = gdimensions
        self.gsites = np.prod(self.gdimensions)
        self.precision = precision
        self.cb = cb
        
        if obj == None:
            self.obj = cgpt.create_grid(gdimensions, precision, cb)
        else:
            self.obj = obj

    def __del__(self):
        cgpt.delete_grid(self.obj)

    def barrier(self):
        cgpt.grid_barrier(self.obj)

    def globalsum(self, x):
        if type(x) == gpt.tensor:
            otype=x.otype
            cgpt.grid_globalsum(self.obj,x.array)
        else:
            return cgpt.grid_globalsum(self.obj,x)
