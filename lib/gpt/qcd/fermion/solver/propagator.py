#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt

class propagator:
    def __init__(self, sc_solver):
        self.sc_solver = sc_solver

    def __call__(self, src, dst):
        grid=src.grid
        # sum_n D^-1 vn vn^dag src = D^-1 vn (src^dag vn)^dag
        dst_sc,src_sc=gpt.vspincolor(grid),gpt.vspincolor(grid)

        for s in range(4):
            for c in range(3):
            
                gpt.qcd.prop_to_ferm(src_sc,src,s,c)

                self.sc_solver(src_sc,dst_sc)

                gpt.qcd.ferm_to_prop(dst,dst_sc,s,c)
