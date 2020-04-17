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
    def dim_mask(nd):
        return [ 0 ] * nd

class redblack:
    n=2
    def dim_mask(nd):
        # four-dimensional red-black
        rbd=min([nd,4])
        return [ 0 ] * (nd-rbd) + [ 1 ] * rbd

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
            fdimensions=[ int(x) for x in p[0].strip("[]").split(",") ]
            precision=gpt.str_to_precision(p[1])
            cb=str_to_checkerboarding(p[2])
        else:
            fdimensions=first
            precision=second
            if third is None:
                cb=full
            else:
                cb=third
            obj=fourth

        self.fdimensions = fdimensions
        self.gsites = np.prod(self.fdimensions)
        self.precision = precision
        self.cb = cb
        self.nd=len(self.fdimensions)
        
        if obj == None:
            self.obj = cgpt.create_grid(fdimensions, precision, cb)
        else:
            self.obj = obj

        # processor is mpi rank, may not be lexicographical (cartesian) rank
        self.processor,self.Nprocessors,self.processor_coor,self.gdimensions,self.ldimensions=cgpt.grid_get_processor(self.obj)
        self.mpi = [ self.gdimensions[i] // self.ldimensions[i] for i in range(self.nd) ]

    def describe(self): # creates a string without spaces that can be used to construct it again, this should only describe the grid geometry not the mpi/simd
        return (str(self.fdimensions)+";"+self.precision.__name__+";"+self.cb.__name__).replace(" ","")

    def cartesian_rank(self):
        rank=0
        for i in reversed(range(self.nd)):
            rank = rank*self.mpi[i] + self.processor_coor[i]
        return rank

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
