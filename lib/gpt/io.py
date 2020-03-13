#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt
import gpt

def load(*a):
    result=[]
    r,metadata=cgpt.load(*a, "io" in gpt.default.verbose)
    for gr in r:
        grid=gpt.grid(gr[1],eval("gpt.precision." + gr[2]),gr[0])
        result_grid=[]
        for t_obj,s_ot,s_pr in gr[3]:
            assert(s_pr == gr[2])
            l=gpt.lattice(grid,eval("gpt.otype." + s_ot),t_obj)
            l.metadata=metadata
            result_grid.append(l)
        result.append(result_grid)
    while len(result) == 1:
        result=result[0]
    return result


