#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import cgpt, gpt, sys
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
        if isinstance(val, gpt.factor) or type(val) in [ gpt.tensor ]:
            self.val = [ (1.0, [ (factor_unary.NONE,val) ]) ]
        elif type(val) == expr:
            self.val = val.val
            unary = unary | val.unary
        elif type(val) == list:
            self.val = val
        elif gpt.util.isnum(val):
            self.val = [ (complex(val), []) ]
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
            # Attempt to close before product to avoid exponential growth of terms.
            # This does not work for sub-expressions without lattice fields, so
            # lhs and rhs may still contain multiple terms.
            if len(lhs.val) > 1:
                lhs=expr(gpt.eval(lhs))
            if len(rhs.val) > 1:
                rhs=expr(gpt.eval(rhs))
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


class factor:

    def __rmul__(self, l):
        return expr(l) * expr(self)

    def __mul__(self, l):
        return expr(self) * expr(l)

    def __truediv__(self, l):
        assert(gpt.util.isnum(l))
        return expr(self) * (1.0/l)

    def __add__(self, l):
        return expr(self) + expr(l)

    def __sub__(self, l):
        return expr(self) - expr(l)

    def __neg__(self):
        return expr(self) * (-1.0)


def get_lattice(e):
    if type(e) == expr:
        assert(len(e.val) > 0)
        return get_lattice(e.val[0][1])
    elif type(e) == list:
        for i in e:
            if type(i[1]) == gpt.lattice:
                return i[1]
    return None

def get_grid(e):
    l=get_lattice(e)
    if l is None:
        return None
    return l.grid

def apply_type_right_to_left(e,t):
    if type(e) == expr:
        return expr([ (x[0],apply_type_right_to_left(x[1],t)) for x in e.val ], e.unary)
    elif type(e) == list:
        n=len(e)
        for i in reversed(range(n)):
            if type(e[i][1]) == t:

                # create operator
                operator=e[i][1].unary(e[i][0])

                # apply operator
                e=e[0:i] + [ (factor_unary.NONE,operator(expr_eval(expr([ (1.0, e[i+1:]) ])))) ]

        return e
    assert(0)

def expr_eval(first, second = None, ac = False):

    # this will always evaluate to a lattice object
    # or remain an expression if it cannot do so

    if not second is None:
        t_obj = first.v_obj
        e = expr(second)
    else:
        assert(ac == False)
        if type(first) == gpt.lattice:
            return first

        e = expr(first)
        lat = get_lattice(e)
        if lat is None:
            # cannot evaluate to a lattice object, leave expression unevaluated
            return first
        grid = lat.grid
        otype = lat.otype
        n = len(otype.v_idx)
        t_obj = None

    # apply matrix_operators
    e = apply_type_right_to_left(e,gpt.matrix_operator)

    # fast return if already a lattice
    if t_obj is None:
        if e.is_single(gpt.lattice):
            ue,uf,v=e.get_single()
            if uf == factor_unary.NONE and ue == expr_unary.NONE:
                return v

    # verbose output
    if gpt.default.is_verbose("eval"):
        gpt.message("GPT::verbose::eval: " + str(e))

    if not t_obj is None:
        for i,t in enumerate(t_obj):
            assert(0 == cgpt.eval(t, e.val, e.unary, ac,i))
        return first
    else:
        assert(ac == False)
        t_obj,s_ot,s_pr=[0]*n,[0]*n,[0]*n
        for i in otype.v_idx:
            t_obj[i],s_ot[i],s_pr[i]=cgpt.eval(t_obj[i], e.val, e.unary, False,i)
        if len(s_ot) == 1:
            otype=eval("gpt.otype." + s_ot[0])
        else:
            otype=gpt.otype.from_v_otype(s_ot)
        return gpt.lattice(grid,otype,t_obj)
