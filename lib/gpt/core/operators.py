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

def apply_expr_unary(l):
    if l.unary == gpt.expr_unary.NONE:
        return l
    return gpt.expr(gpt.eval(l))

def trace(l, t = None):
    if t is None:
        t = gpt.expr_unary.BIT_SPINTRACE|gpt.expr_unary.BIT_COLORTRACE
    if type(l) == gpt.tensor:
        return l.trace(t)
    return gpt.expr( l, t )

def expr_eval(first, second = None, ac = False):

    if not second is None:
        t_obj = first.obj
        e = gpt.expr(second)
    else:
        if type(first) == gpt.lattice:
            return first

        e = gpt.expr(first)
        t_obj = 0

    if gpt.default.is_verbose("eval"):
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
    return gpt.util.value_to_tensor( cgpt.lattice_sum(l.obj), l.otype )
