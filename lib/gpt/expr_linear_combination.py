#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt
import gpt

class expr_linear_combination:
    def __init__(self, val):
        self.val = val

    def __mul__(self, l):
        if (gpt.util.isnum(l)):
            return expr_linear_combination([ (a[0]*l,a[1]) for a in self.val ])
        elif type(l) == gpt.lattice:
            raise Exception("The current order may be inefficient, please bracket lattice factors first")
        else:
            raise Exception("Unkown factor")

    def __rmul__(self, l):
        if (gpt.util.isnum(l)):
            return expr_linear_combination([ (a[0]*l,a[1]) for a in self.val ])
        elif type(l) == gpt.lattice:
            raise Exception("The current order may be inefficient, please bracket lattice factors first")
        else:
            raise Exception("Unkown factor")

    def __truediv__(self, l):
        assert(gpt.util.isnum(l))
        return expr_linear_combination([ (a[0]/l,a[1]) for a in self.val ])
    
    def __add__(self, l):
        if type(l) == gpt.lattice:
            return expr_linear_combination(self.val + [(1,l)])
        elif type(l) == gpt.expr_linear_combination:
            return expr_linear_combination(self.val + l.val)
        else:
            raise Exception("Unknown combination")

    def __sub__(self, l):
        return self.__add__(l.__neg__())

    def __neg__(self):
        return gpt.expr_linear_combination([ (-a[0],a[1]) for a in self.val ])

def eval(first, second = None):
    if type(first) == gpt.lattice:
        t = first
        e = second
        assert(len(e.val) > 0)
    else:
        e = first
        assert(len(e.val) > 0)
        t = gpt.lattice(e.val[0][1])

    cgpt.eval(t.obj, [ (a[0],a[1].obj) for a in e.val ])
    return t
