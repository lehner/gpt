#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt
class grid:
    def __init__(self, gdimensions, precision):
        print("Create grid with gdimensions = %s and precision = %s" % (str(gdimensions),str(precision)))
        self.obj = cgpt.create_grid(gdimensions, precision)

    def __del__(self):
        cgpt.delete(self.obj)


