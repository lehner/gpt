#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt
import gpt
import numpy as np

class grid:
    def __init__(self, gdimensions, precision, obj = None):
        self.gdimensions = gdimensions
        self.gsites = np.prod(self.gdimensions)
        self.precision = precision
        if obj == None:
            self.obj = cgpt.create_grid(gdimensions, precision)
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
