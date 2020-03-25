#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt

class eo_ne:
    def __init__(self, matrix, inverter):
        self.matrix = matrix
        self.inverter = inverter

        self.F_grid_eo=matrix.F_grid_eo
        self.F_grid=matrix.F_grid

        self.ie=gpt.vspincolor(self.F_grid_eo)
        self.io=gpt.vspincolor(self.F_grid_eo)
        self.t1=gpt.vspincolor(self.F_grid_eo)
        self.t2=gpt.vspincolor(self.F_grid_eo)
        self.oe=gpt.vspincolor(self.F_grid_eo)
        self.oo=gpt.vspincolor(self.F_grid_eo)
        self.ftmp=gpt.vspincolor(self.F_grid)

    def __call__(self, src_sc, dst_sc):

        self.matrix.import_physical(src_sc, self.ftmp)

        gpt.pick_cb(gpt.even,self.ie,self.ftmp)
        gpt.pick_cb(gpt.odd,self.io,self.ftmp)

        # D^-1 = L NDagN^-1 R + S

        self.matrix.R(self.ie, self.io, self.t1)

        self.t2[:]=0
        gpt.change_cb(self.t2,gpt.even)

        self.inverter(lambda i,o: self.matrix.NDagN(i,o),self.t1,self.t2)

        self.matrix.L(self.t2, self.oe, self.oo)

        self.matrix.S(self.ie,self.io,self.t1,self.t2)

        self.oe += self.t1
        self.oo += self.t2

        gpt.set_cb(self.ftmp,self.oe)
        gpt.set_cb(self.ftmp,self.oo)

        self.matrix.export_physical(self.ftmp,dst_sc)

