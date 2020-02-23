#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt
class grid:
    def __init__(self, gdimensions, precision):
        self.gdimensions = gdimensions
        self.precision = precision
        self.obj = cgpt.create_grid(gdimensions, precision)

    def __del__(self):
        cgpt.delete_grid(self.obj)

    def barrier(self):
        cgpt.grid_barrier(self.obj)
