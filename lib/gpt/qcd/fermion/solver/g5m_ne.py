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

    def __call__(self, src_sc, dst_sc):
        dst_sc @= gpt.gamma[5] * src_sc
        self.matrix.G5M(dst_sc,src_sc)
        
        dst_sc[:]=0
        
        tmp=gpt.lattice(src_sc)
        
        self.inverter(lambda i,o: (self.matrix.G5M(i,tmp),self.matrix.G5M(tmp,o)),src_sc,dst_sc)
