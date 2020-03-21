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

        self.grid_eo = matrix.grid_eo
        self.ie=gpt.vspincolor(self.grid_eo)
        self.io=gpt.vspincolor(self.grid_eo)
        self.t1=gpt.vspincolor(self.grid_eo)
        self.t2=gpt.vspincolor(self.grid_eo)
        self.oe=gpt.vspincolor(self.grid_eo)
        self.oo=gpt.vspincolor(self.grid_eo)

    def __call__(self, src_sc, dst_sc):

        gpt.pick_cb(gpt.even,self.ie,src_sc)
        gpt.pick_cb(gpt.odd,self.io,src_sc)

        # D^-1 = L NDagN^-1 R + S

        self.matrix.R(self.ie, self.io, self.t1)

        self.t2[:]=0
        gpt.change_cb(self.t2,gpt.even)

        self.inverter(lambda i,o: self.matrix.NDagN(i,o),self.t1,self.t2)

        self.matrix.L(self.t2, self.oe, self.oo)

        self.matrix.S(self.ie,self.io,self.t1,self.t2)

        self.oe += self.t1
        self.oo += self.t2

        gpt.set_cb(dst_sc,self.oe)
        gpt.set_cb(dst_sc,self.oo)

