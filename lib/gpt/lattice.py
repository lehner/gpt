#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt
class lattice:
    def __init__(self, grid, otype):
        self.grid = grid
        self.otype = otype
        self.obj = cgpt.create_lattice(self.grid.obj, self.otype, self.grid.precision)

    def __del__(self):
        cgpt.delete_lattice(self.obj)

    def __setitem__(self, key, value):
        if key == slice(None,None,None):
            key = ()
        
        assert(type(key) == tuple)
        cgpt.lattice_set_val(self.obj, key, value)

    def to_dict(self):
        return cgpt.lattice_to_dict(self.obj)

    def __str__(self):
        return str(self.to_dict())
