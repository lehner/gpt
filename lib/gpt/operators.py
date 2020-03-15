#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt
import gpt
import numpy as np

def get_grid(e):
    if type(e) == gpt.expr:
        assert(len(e.val) > 0)
        return get_grid(e.val[0][1])
    elif type(e) == list:
        for i in e:
            if type(i[1]) == gpt.lattice:
                return i[1].grid
        assert(0) # should never happen for a properly formed expression
    else:
        assert(0)

def conj(l):
    if type(l) == gpt.expr:
        return gpt.expr( [ (complex(a[0]).conjugate(),[ (x[0] ^ (gpt.factor_unary.BIT_CONJ),x[1]) for x in a[1] ]) for a in l.val ] )
    elif type(l) == gpt.tensor:
        return l.conj()
    else:
        return conj(gpt.expr(l))

def transpose(l):
    if type(l) == gpt.expr:
        return gpt.expr( [ (a[0],[ (x[0] ^ (gpt.factor_unary.BIT_TRANS),x[1]) for x in reversed(a[1]) ]) for a in l.val ] )
    elif type(l) == gpt.tensor and l.transposable():
        return l.transpose()
    else:
        return transpose(gpt.expr(l))

def adj(l):
    if type(l) == gpt.expr:
        return gpt.expr( [ (complex(a[0]).conjugate(),[ (x[0] ^ (gpt.factor_unary.BIT_TRANS|gpt.factor_unary.BIT_CONJ),x[1]) for x in reversed(a[1]) ]) for a in l.val ] )
    elif type(l) == gpt.tensor and l.transposable():
        return l.adj()
    else:
        return adj(gpt.expr(l))

def trace(l, t = gpt.expr_unary.BIT_SPINTRACE|gpt.expr_unary.BIT_COLORTRACE):
    if type(l) == gpt.tensor:
        return l.trace(t)
    return gpt.expr( l, t )

def apply_expr_unary(l):
    if l.unary == gpt.expr_unary.NONE:
        return l
    return gpt.expr(gpt.eval(l))

def expr_eval(first, second = None, ac = False):

    if not second is None:
        t_obj = first.obj
        e = gpt.expr(second)
    else:
        if type(first) == gpt.lattice:
            return first

        e = gpt.expr(first)
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

def sum(e):
    l=gpt.eval(e)
    val= cgpt.lattice_sum(l.obj)
    if type(val) == complex:
        return val
    return gpt.tensor(val, l.otype)
