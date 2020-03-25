#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt
import cgpt

class operator:
    def __init__(self, name, U, params, Ls = None):
        self.name = name
        self.U = U
        self.U_grid = U[0].grid
        self.U_grid_eo = gpt.grid(self.U_grid.gdimensions,self.U_grid.precision,gpt.redblack)
        if Ls is None:
            self.F_grid = self.U_grid
            self.F_grid_eo = self.U_grid_eo
        else:
            self.F_grid = gpt.grid(self.U_grid.gdimensions + [ Ls ],self.U_grid.precision)
            self.F_grid_eo = gpt.grid(self.F_grid.gdimensions,self.U_grid.precision,gpt.redblack)

        self.params = {
            "U_grid" : self.U_grid.obj,
            "U_grid_rb" : self.U_grid_eo.obj,
            "F_grid" : self.F_grid.obj,
            "F_grid_rb" : self.F_grid_eo.obj,
            "U" : [ u.obj for u in self.U ]
        }

        for k in params:
            assert(not k in [ "U_grid", "U_grid_rb", "F_grid", "F_grid_rb", "U" ])
            self.params[k] = params[k]

        self.obj = cgpt.create_fermion_operator(name,self.U_grid.precision,self.params)

        # register matrix operations
        gpt.qcd.fermion.register(self)

    def __del__(self):
        cgpt.delete_fermion_operator(self.obj)

    def unary(self, opcode, i, o):
        return cgpt.apply_fermion_operator(self.obj,opcode,i.obj,o.obj)

    def G5M(self, i, o):
        self.M(i,o)
        o @= gpt.gamma[5] * o

    
