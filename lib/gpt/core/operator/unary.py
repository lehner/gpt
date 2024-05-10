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


def conj(l):
    if isinstance(l, gpt.expr):
        return gpt.expr(
            [
                (
                    complex(a[0]).conjugate(),
                    [(x[0] ^ (gpt.factor_unary.BIT_CONJ), x[1]) for x in a[1]],
                )
                for a in l.val
            ]
        )
    elif isinstance(l, gpt.tensor):
        return l.conj()
    else:
        return conj(gpt.expr(l))


def transpose(l):
    if isinstance(l, gpt.expr):
        return gpt.expr(
            [
                (
                    a[0],
                    [(x[0] ^ (gpt.factor_unary.BIT_TRANS), x[1]) for x in reversed(a[1])],
                )
                for a in l.val
            ]
        )
    elif isinstance(l, gpt.tensor) and l.transposable():
        return l.transpose()
    else:
        return transpose(gpt.expr(l))


def adj(l):
    if isinstance(l, gpt.expr):
        return gpt.expr(
            [
                (
                    complex(a[0]).conjugate(),
                    [
                        (
                            x[0] ^ (gpt.factor_unary.BIT_TRANS | gpt.factor_unary.BIT_CONJ),
                            x[1],
                        )
                        for x in reversed(a[1])
                    ],
                )
                for a in l.val
            ]
        )
    elif isinstance(l, gpt.matrix_operator):
        return l.adj()
    elif isinstance(l, gpt.core.foundation.base):
        return l.__class__.foundation.adj(l)
    elif gpt.util.is_num(l):
        return gpt.util.adj_num(l)
    else:
        return adj(gpt.expr(l))


def inv(l):
    if isinstance(l, gpt.matrix_operator):
        return l.inv()
    else:
        assert 0


def apply_expr_unary(l):
    if l.unary == gpt.expr_unary.NONE:
        return l
    return gpt.expr(gpt.eval(l))


def trace(l, t=None):
    if t is None:
        t = gpt.expr_unary.BIT_SPINTRACE | gpt.expr_unary.BIT_COLORTRACE
    if isinstance(l, gpt.core.foundation.base):
        return l.__class__.foundation.trace(l, t)
    elif gpt.util.is_num(l):
        return l
    else:
        return gpt.expr(l, t)


def spin_trace(l):
    return trace(l, gpt.expr_unary.BIT_SPINTRACE)


def color_trace(l):
    return trace(l, gpt.expr_unary.BIT_COLORTRACE)


def rank_sum(e):
    if isinstance(e, gpt.expr):
        e = gpt.eval(e)
    return e.__class__.foundation.rank_sum(e)


def sum(e):
    if isinstance(e, gpt.expr):
        e = gpt.eval(e)
    return e.__class__.foundation.sum(e)
