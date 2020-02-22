#
# GPT
#
# Authors: Christoph Lehner 2020
#
class lattice:
    def __init__(self, grid, otype):
        self.grid = grid
        self.otype = otype

    def __setitem__(self, *key):
        print(key)

    def __str__(self):
        return str(self.otype)

