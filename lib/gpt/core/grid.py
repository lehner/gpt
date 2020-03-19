#
# GPT
#
# Authors: Christoph Lehner 2020
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
