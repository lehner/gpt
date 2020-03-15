#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt
import gpt
import numpy as np

class factor_unary:
    NONE = 0
    BIT_TRANS = 1
    BIT_CONJ = 2

class expr_unary:
    NONE = 0
    BIT_SPINTRACE = 1
    BIT_COLORTRACE = 2

# expr:
# - each expression can have a unary operation such as trace
# - each expression has linear combination of terms
# - each term is a non-commutative product of factors
# - each factor is a lattice/object with optional factor_unary operation applied
# - an object could be a spin or a gauge matrix

class expr:
    def __init__(self, val, unary = expr_unary.NONE):
        if type(val) == gpt.lattice:
            self.val = [ (1.0, [ (factor_unary.NONE,val) ]) ]
        #elif type(val) == gpt.gamma:
        #    self.val = [ (1.0, [ (factor_unary.NONE,val) ]) ]
        elif type(val) == gpt.tensor:
            self.val = [ (1.0, [ (factor_unary.NONE,val) ]) ]
        elif type(val) == expr:
            self.val = val.val
            unary = unary | val.unary
        elif type(val) == list:
            self.val = val
        elif gpt.util.isnum(val):
            self.val = [ (val, []) ]
        else:
            raise Exception("Unknown type " + str(type(val)))
        self.unary = unary

    def is_single(self, t = None):
        b=(len(self.val) == 1 and self.val[0][0] == 1.0 and
           len(self.val[0][1]) == 1)
        if not t is None:
            b = b and type(self.val[0][1][0][1]) == t
        return b

    def get_single(self):
        return (self.unary, self.val[0][1][0][0],self.val[0][1][0][1])

    def __mul__(self, l):
        if type(l) == expr:
            lhs = gpt.apply_expr_unary(self)
            rhs = gpt.apply_expr_unary(l)
            # close before product to avoid exponential growth of terms
            if len(lhs.val) > 1:
                lhs=expr(gpt.eval(lhs))
            if len(rhs.val) > 1:
                rhs=expr(gpt.eval(rhs))
            assert(len(lhs.val) == 1 or len(rhs.val) == 1)
            return expr( [ (a[0]*b[0], a[1] + b[1]) for a in lhs.val for b in rhs.val ] )
        elif type(l) == gpt.tensor and self.is_single(gpt.tensor):
            ue,uf,to=self.get_single()
            if ue == 0 and uf & factor_unary.BIT_TRANS != 0:
                tag = (to.otype,l.otype)
                assert(tag in gpt.otype.itab)
                mt=gpt.otype.itab[tag]
                lhs=to.array
                if uf & gpt.factor_unary.BIT_CONJ != 0:
                    lhs=lhs.conj()
                res=gpt.tensor( np.tensordot(lhs, l.array, axes = mt[1]), mt[0])
                if res.otype == gpt.ot_complex:
                    res = complex(res.array)
                return res
            assert(0)
        else:
            return self.__mul__(expr(l))

    def __rmul__(self, l):
        if type(l) == expr:
            return l.__mul__(self)
        else:
            return self.__rmul__(expr(l))

    def __truediv__(self, l):
        if not gpt.util.isnum(l):
            raise Exception("At this point can only divide by numbers")
        return self.__mul__(expr(1.0/l))
    
    def __add__(self, l):
        if type(l) == expr:
            if self.unary == l.unary:
                return expr( self.val + l.val, self.unary )
            else:
                return expr( gpt.apply_expr_unary(self).val + gpt.apply_expr_unary(l).val )
        else:
            return self.__add__(expr(l))

    def __sub__(self, l):
        return self.__add__(l.__neg__())

    def __neg__(self):
        return expr( [ (-a[0],a[1]) for a in self.val ], self.unary )

    def __str__(self):
        ret=""

        if self.unary & expr_unary.BIT_SPINTRACE:
            ret=ret + "spinTrace("
        if self.unary & expr_unary.BIT_COLORTRACE:
            ret=ret + "colorTrace("

        for t in self.val:
            ret=ret + " + (" + str(t[0]) + ")"
            for f in t[1]:
                ret = ret + "*"
                if f[0] == factor_unary.NONE:
                    ret = ret + repr(f[1])
                elif f[0] == factor_unary.BIT_CONJ|factor_unary.BIT_TRANS:
                    ret = ret + "adj(" + repr(f[1]) + ")"
                elif f[0] == factor_unary.BIT_CONJ:
                    ret = ret + "conjugate(" + repr(f[1]) + ")"
                elif f[0] == factor_unary.BIT_TRANS:
                    ret = ret + "transpose(" + repr(f[1]) + ")"
                else:
                    ret = ret + "??"

        if self.unary & expr_unary.BIT_SPINTRACE:
            ret=ret + ")"
        if self.unary & expr_unary.BIT_COLORTRACE:
            ret=ret + ")"
        return ret
