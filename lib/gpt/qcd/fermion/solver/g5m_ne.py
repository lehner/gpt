#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt

class g5m_ne:
    def __init__(self, matrix, inverter):
        self.matrix = matrix
        self.inverter = inverter
        self.F_grid=matrix.F_grid
        self.ftmp=gpt.vspincolor(self.F_grid)
        self.ftmp2=gpt.vspincolor(self.F_grid)
        self.ftmp3=gpt.vspincolor(self.F_grid)

    def __call__(self, src_sc, dst_sc):

        self.matrix.ImportPhysicalFermionSource(src_sc, self.ftmp)

        self.ftmp @= gpt.gamma[5] * self.ftmp
        self.matrix.G5M(self.ftmp,self.ftmp2)
        
        self.ftmp[:]=0
        self.inverter(lambda i,o: (self.matrix.G5M(i,self.ftmp3),self.matrix.G5M(self.ftmp3,o)),self.ftmp2,self.ftmp)

        self.matrix.ExportPhysicalFermionSolution(self.ftmp,dst_sc)
