#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt
import gpt

class field_unary:
    NONE = 0
    BIT_TRANS = 1
    BIT_CONJ = 2
    ADJ = 3
    CONJ = 2
    TRANS = 1

class expr_unary:
    NONE = 0
    BIT_SPINTRACE = 1
    BIT_COLORTRACE = 2

# expr:
# - can have global unary operation such as trace
# - has linear combination of terms
# - each term is a non-commutative product of field_expr
# - each field_expr is a lattice with optional field_unary operation applied

class expr:
    def __init__(self, val, unary = expr_unary.NONE):
        if type(val) == gpt.lattice:
            self.val = [ (1.0, [ (field_unary.NONE,val) ]) ]
        elif type(val) == expr:
            self.val = val.val
        elif type(val) == list:
            self.val = val
        elif gpt.util.isnum(val):
            self.val = [ (val, []) ]
        else:
            raise Exception("Unknown type " + str(type(val)))
        self.unary = unary

    def __mul__(self, l):
        if type(l) == expr:
            l = apply_unary(l)
            return expr( [ (a[0]*b[0],a[1] + b[1]) for a in self.val for b in l.val ], self.unary )
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
                return expr( self.val + l.val , self.unary )
            else:
                return expr( apply_unary(self) + apply_unary(l), expr_unary.NONE )
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
                if f[0] == field_unary.NONE:
                    ret = ret + repr(f[1])
                elif f[0] == field_unary.ADJ:
                    ret = ret + "adj(" + repr(f[1]) + ")"
                elif f[0] == field_unary.CONJ:
                    ret = ret + "conjugate(" + repr(f[1]) + ")"
                elif f[0] == field_unary.TRANS:
                    ret = ret + "transpose(" + repr(f[1]) + ")"
                else:
                    ret = ret + "??"
        if self.unary & expr_unary.BIT_SPINTRACE:
            ret=ret + ")"
        if self.unary & expr_unary.BIT_COLORTRACE:
            ret=ret + ")"
        return ret

    def lattice(self):
        if (len(self.val) == 0 or len(self.val[0][1]) == 0):
            raise Exception("Expression's lattice type is yet undefined")
        lat=self.val[0][1][0][1]
        grid=lat.grid
        otype=lat.otype
        if self.unary & expr_unary.BIT_SPINTRACE:
            otype=otype.SPINTRACE_OTYPE
        if self.unary & expr_unary.BIT_COLORTRACE:
            otype=otype.COLORTRACE_OTYPE
        return gpt.lattice(grid,otype)

def adj(l):
    if type(l) == expr:
        return expr( [ (complex(a[0]).conjugate(),[ (x[0] ^ (field_unary.BIT_TRANS|field_unary.BIT_CONJ),x[1]) for x in reversed(a[1]) ]) for a in l.val ] )
    else:
        return adj(expr(l))

def conj(l):
    if type(l) == expr:
        return expr( [ (complex(a[0]).conjugate(),[ (x[0] ^ (field_unary.BIT_CONJ),x[1]) for x in a[1] ]) for a in l.val ] )
    else:
        return adj(expr(l))

def transpose(l):
    if type(l) == expr:
        return expr( [ (a[0],[ (x[0] ^ (field_unary.BIT_TRANS),x[1]) for x in reversed(a[1]) ]) for a in l.val ] )
    else:
        return adj(expr(l))

def trace(l, t = expr_unary.BIT_SPINTRACE|expr_unary.BIT_COLORTRACE):
    if type(l) == expr:
        return expr( l.val, l.unary | t )
    else:
        return expr( l, t )

def apply_unary(l):
    if l.unary == expr_unary.NONE:
        return l
    return gpt.eval(l)

def eval(first, second = None):
    if not second is None:
        t = first
        e = second
        assert(len(e.val) > 0)
    else:
        e = first
        t = e.lattice()

    cgpt.eval(t.obj, e.val, e.unary)
    return t
