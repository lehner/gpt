#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt as g
cg=g.algorithms.iterative.cg

class g5m:
    def __init__(self, eps, maxiter):
        self.eps = eps
        self.maxiter = maxiter

    def __call__(self, w, dst, src):

        grid=src.grid
        # sum_n D^-1 vn vn^dag src = D^-1 vn (src^dag vn)^dag
        dst_sc,src_sc=g.vspincolor(grid),g.vspincolor(grid)
        dst[:]=0

        for s in range(4):
            for c in range(3):

                g.qcd.prop_to_ferm(src_sc,src,s,c)

                dst_sc @= g.gamma[5] * src_sc
                w.G5M(src_sc,dst_sc)        

                dst_sc[:]=0

                cg(lambda i,o: w.G5Msqr(o,i),src_sc,dst_sc,self.eps,self.maxiter)
        
                g.qcd.ferm_to_prop(dst,dst_sc,s,c)
