#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt
import cgpt

class operator:
    def __init__(self, name, U, params):
        self.name = name
        self.U = U
        self.grid = U[0].grid
        self.grid_eo = gpt.grid(self.grid.gdimensions,self.grid.precision,gpt.redblack)
        self.params = {
            "grid" : self.grid.obj,
            "grid_rb" : self.grid_eo.obj,
            "U" : [ u.obj for u in self.U ]
        }
        for k in params:
            assert(not k in [ "grid", "grid_rb", "U" ])
            self.params[k] = params[k]
        self.obj = cgpt.create_fermion_operator(name,self.grid.precision,self.params)

        # register matrix operations
        gpt.qcd.fermion.register(self)

    def __del__(self):
        cgpt.delete_fermion_operator(self.obj)

    def unary(self, opcode, i, o):
        return cgpt.apply_fermion_operator(self.obj,opcode,i.obj,o.obj)
