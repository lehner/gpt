#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt
import gpt

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

    def __mul__(self, l):
        if type(l) == expr:
            lhs = apply_unary(self)
            rhs = apply_unary(l)
            
            # need to avoid exponential growth of terms, so close before multiplying
            # strategy: close the expr with more terms since it likely can be evaluated
            #           more efficiently

            # TODO: maybe even close both?
            if len(lhs.val) > 1 and len(rhs.val) > 1:
                if len(lhs.val) >= len(rhs.val):
                    lhs=expr(gpt.eval(lhs))
                else:
                    rhs=expr(gpt.eval(rhs))
            assert(len(lhs.val) == 1 or len(rhs.val) == 1)
            return expr( [ (a[0]*b[0], a[1] + b[1]) for a in lhs.val for b in rhs.val ] )
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
                return expr( apply_unary(self).val + apply_unary(l).val )
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

def get_grid(e):
    if type(e) == expr:
        assert(len(e.val) > 0)
        return get_grid(e.val[0][1])
    elif type(e) == list:
        for i in e:
            if type(i[1]) == gpt.lattice:
                return i[1].grid
        assert(0) # should never happen for a properly formed expression
    else:
        assert(0)

def adj(l):
    if type(l) == expr:
        return expr( [ (complex(a[0]).conjugate(),[ (x[0] ^ (factor_unary.BIT_TRANS|factor_unary.BIT_CONJ),x[1]) for x in reversed(a[1]) ]) for a in l.val ] )
    else:
        return adj(expr(l))

def conj(l):
    if type(l) == expr:
        return expr( [ (complex(a[0]).conjugate(),[ (x[0] ^ (factor_unary.BIT_CONJ),x[1]) for x in a[1] ]) for a in l.val ] )
    else:
        return adj(expr(l))

def transpose(l):
    if type(l) == expr:
        return expr( [ (a[0],[ (x[0] ^ (factor_unary.BIT_TRANS),x[1]) for x in reversed(a[1]) ]) for a in l.val ] )
    else:
        return adj(expr(l))

def trace(l, t = expr_unary.BIT_SPINTRACE|expr_unary.BIT_COLORTRACE):
    return expr( l, t )

def apply_unary(l):
    if l.unary == expr_unary.NONE:
        return l
    return expr(gpt.eval(l))

def expr_eval(first, second = None, ac = False):
    if not second is None:
        t_obj = first.obj
        e = second
    else:
        e = first
        t_obj = 0

    if "eval" in gpt.default.verbose:
        gpt.message("GPT::verbose::eval: " + str(e))

    if t_obj != 0:
        assert(0 == cgpt.eval(t_obj, e.val, e.unary, ac))
        return first
    else:
        assert(ac == False)
        t_obj,s_ot,s_pr=cgpt.eval(t_obj, e.val, e.unary, False)
        grid=get_grid(e)
        return gpt.lattice(grid,eval("gpt.otype." + s_ot),t_obj)
