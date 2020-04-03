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

def str_to_checkerboarding(s):
    if s == "full":
        return full
    elif s == "redblack":
        return redblack
    else:
        assert(0)

class grid:
    def __init__(self, first, second = None, third = None, fourth = None):
        if type(first) == str:
            # create from description
            p=first.split(";")
            gdimensions=[ int(x) for x in p[0].strip("[]").split(",") ]
            precision=gpt.str_to_precision(p[1])
            cb=str_to_checkerboarding(p[2])
        else:
            gdimensions=first
            precision=second
            if third is None:
                cb=full
            else:
                cb=third
            obj=fourth

        self.gdimensions = gdimensions
        self.gsites = np.prod(self.gdimensions)
        self.precision = precision
        self.cb = cb
        
        if obj == None:
            self.obj = cgpt.create_grid(gdimensions, precision, cb)
        else:
            self.obj = obj

        self.processor,self.Nprocessors,self.processor_coor,self.ldimensions=cgpt.grid_get_processor(self.obj)

    def describe(self): # creates a string without spaces that can be used to construct it again, this should only describe the grid geometry not the mpi/simd
        return (str(self.gdimensions)+";"+self.precision.__name__+";"+self.cb.__name__).replace(" ","")

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
