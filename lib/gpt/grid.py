#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt
import numpy as np

class grid:
    def __init__(self, gdimensions, precision):
        self.gdimensions = gdimensions
        self.gsites = np.prod(self.gdimensions)
        self.precision = precision
        self.obj = cgpt.create_grid(gdimensions, precision)

    def __del__(self):
        cgpt.delete_grid(self.obj)

    def barrier(self):
        cgpt.grid_barrier(self.obj)

    def globalsum(self, x):
        return cgpt.grid_globalsum(self.obj,x)
