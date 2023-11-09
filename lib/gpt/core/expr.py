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
from gpt.default import is_verbose

verbose = is_verbose("eval")
verbose_performance = is_verbose("eval_performance")


class factor_unary:
    NONE = 0
    BIT_TRANS = 1
    BIT_CONJ = 2
    ADJ = 3


class expr_unary:
    NONE = 0
    BIT_SPINTRACE = 1
    BIT_COLORTRACE = 2


# expr:
# - each expression can have a unary operation such as trace
# - each expression has linear combination of terms
# - each term is a non-commutative product of factors
# - each factor is a list of lattices/objects with optional factor_unary operation applied
# - an object could be a spin or a gauge matrix


class expr:
    def __init__(self, val, unary=expr_unary.NONE):
        if isinstance(val, (gpt.factor, gpt.tensor)):
            self.val = [(1.0, [(factor_unary.NONE, val)])]
        elif isinstance(val, expr):
            self.val = val.val
            unary = unary | val.unary
        elif isinstance(val, list):
            if isinstance(val[0], tuple):
                self.val = val
            else:
                self.val = [(1.0, [(factor_unary.NONE, val)])]
        elif gpt.util.is_num(val):
            self.val = [(complex(val), [])]
        elif val is None:
            self.val = []
        else:
            raise Exception("Unknown type " + str(type(val)))
        self.unary = unary

    def is_single(self, t=None):
        b = len(self.val) == 1 and self.val[0][0] == 1.0 and len(self.val[0][1]) == 1
        if t is not None:
            b = b and gpt.util.is_list_instance(self.val[0][1][0][1], t)
        return b

    def get_single(self):
        return (
            self.unary,
            self.val[0][1][0][0],
            self.val[0][1][0][1],
        )

    def is_num(self):
        return len(self.val) == 1 and len(self.val[0][1]) == 0

    def get_num(self):
        return self.val[0][0]

    def lattice(self):
        for v in self.val:
            for i in v[1]:
                if gpt.util.is_list_instance(i[1], gpt.lattice):
                    return i[1]
        return None

    def __mul__(self, l):
        if isinstance(l, expr):
            lhs = gpt.apply_expr_unary(self)
            rhs = gpt.apply_expr_unary(l)
            # Attempt to close before product to avoid exponential growth of terms.
            # This does not work for sub-expressions without lattice fields, so
            # lhs and rhs may still contain multiple terms.
            if len(lhs.val) > 1:
                lhs = expr(gpt.eval(lhs))
            if len(rhs.val) > 1:
                rhs = expr(gpt.eval(rhs))
            return expr([(a[0] * b[0], a[1] + b[1]) for a in lhs.val for b in rhs.val])
        elif isinstance(l, gpt.tensor) and self.is_single(gpt.tensor):
            ue, uf, to = self.get_single()
            if ue == 0 and uf & factor_unary.BIT_TRANS != 0:
                tag = l.otype.__name__
                assert tag in to.otype.itab
                mt = to.otype.itab[tag]
                lhs = to.array
                if uf & gpt.factor_unary.BIT_CONJ != 0:
                    lhs = lhs.conj()
                res = gpt.tensor(np.tensordot(lhs, l.array, axes=mt[1]), mt[0]())
                if res.otype == gpt.ot_singlet:
                    res = complex(res.array)
                return res
            assert 0
        else:
            return self.__mul__(expr(l))

    def __rmul__(self, l):
        if isinstance(l, expr):
            return l.__mul__(self)
        else:
            return self.__rmul__(expr(l))

    def __truediv__(self, l):
        if gpt.util.is_num(l) is False:
            raise Exception("At this point can only divide by numbers")
        return self.__mul__(expr(1.0 / l))

    def __add__(self, l):
        if isinstance(l, expr):
            if self.unary == l.unary:
                return expr(self.val + l.val, self.unary)
            else:
                return expr(gpt.apply_expr_unary(self).val + gpt.apply_expr_unary(l).val)
        else:
            return self.__add__(expr(l))

    def __sub__(self, l):
        return self.__add__(l.__neg__())

    def __neg__(self):
        return expr([(-a[0], a[1]) for a in self.val], self.unary)

    def __str__(self):
        ret = ""

        if self.unary & expr_unary.BIT_SPINTRACE:
            ret = ret + "spinTrace("
        if self.unary & expr_unary.BIT_COLORTRACE:
            ret = ret + "colorTrace("

        for t in self.val:
            ret = ret + " + (" + str(t[0]) + ")"
            for f in t[1]:
                ret = ret + "*"
                if f[0] == factor_unary.NONE:
                    ret = ret + repr(f[1])
                elif f[0] == factor_unary.ADJ:
                    ret = ret + "adj(" + repr(f[1]) + ")"
                elif f[0] == factor_unary.BIT_CONJ:
                    ret = ret + "conjugate(" + repr(f[1]) + ")"
                elif f[0] == factor_unary.BIT_TRANS:
                    ret = ret + "transpose(" + repr(f[1]) + ")"
                else:
                    ret = ret + "??"

        if self.unary & expr_unary.BIT_SPINTRACE:
            ret = ret + ")"
        if self.unary & expr_unary.BIT_COLORTRACE:
            ret = ret + ")"
        return ret


class factor:
    def __rmul__(self, l):
        return expr(l) * expr(self)

    def __mul__(self, l):
        return expr(self) * expr(l)

    def __truediv__(self, l):
        assert gpt.util.is_num(l)
        return expr(self) * (1.0 / l)

    def __add__(self, l):
        return expr(self) + expr(l)

    def __sub__(self, l):
        return expr(self) - expr(l)

    def __neg__(self):
        return expr(self) * (-1.0)


def apply_type_right_to_left(e, t):
    if isinstance(e, expr):
        return expr([(x[0], apply_type_right_to_left(x[1], t)) for x in e.val], e.unary)
    elif isinstance(e, list):
        n = len(e)
        for i in reversed(range(n)):
            if isinstance(e[i][1], t):
                # create operator
                operator = e[i][1].unary(e[i][0])

                # apply operator
                e = e[0:i] + [(factor_unary.NONE, operator(expr_eval(expr([(1.0, e[i + 1 :])]))))]

        return e
    assert 0


def get_otype_from_multiplication(t_otype, t_adj, f_otype, f_adj):
    if f_adj and not t_adj and f_otype.itab is not None:
        # inner
        tab = f_otype.itab
        rtab = {}
    elif t_adj and not f_adj and f_otype.otab is not None:
        # outer
        tab = f_otype.otab
        rtab = {}
    else:
        tab = f_otype.mtab
        rtab = t_otype.rmtab

    if t_otype.__name__ in tab:
        return tab[t_otype.__name__][0]()
    else:
        if f_otype.__name__ not in rtab:
            if f_otype.data_alias is not None:
                return get_otype_from_multiplication(t_otype, t_adj, f_otype.data_alias(), f_adj)
            elif t_otype.data_alias is not None:
                return get_otype_from_multiplication(t_otype.data_alias(), t_adj, f_otype, f_adj)
            else:
                gpt.message(
                    "Missing entry in multiplication table: %s x %s"
                    % (t_otype.__name__, f_otype.__name__)
                )
        return rtab[f_otype.__name__][0]()


def get_otype_from_expression(e):
    bare_otype = None
    for coef, term in e.val:
        if len(term) == 0:
            t_otype = gpt.ot_singlet
        else:
            t_otype = None
            t_adj = False
            for unary, factor in reversed(term):
                f_otype = gpt.util.to_list(factor)[0].otype
                f_adj = unary == factor_unary.ADJ
                if t_otype is None:
                    t_otype = f_otype
                    t_adj = f_adj
                else:
                    t_otype = get_otype_from_multiplication(t_otype, t_adj, f_otype, f_adj)

        if bare_otype is None:
            bare_otype = t_otype
        else:
            # all elements of a sum must have same data type
            assert t_otype.data_otype().__name__ == bare_otype.data_otype().__name__

    # apply unaries
    if e.unary & expr_unary.BIT_SPINTRACE:
        st = bare_otype.spintrace
        assert st is not None
        if st[2] is not None:
            bare_otype = st[2]()
    if e.unary & expr_unary.BIT_COLORTRACE:
        ct = bare_otype.colortrace
        assert ct is not None
        if ct[2] is not None:
            bare_otype = ct[2]()
    return bare_otype


def expr_eval(first, second=None, ac=False):
    t = gpt.timer("eval", verbose_performance)

    # this will always evaluate to a (list of) lattice object(s)
    # or remain an expression if it cannot do so

    t("prepare")
    if second is not None:
        dst = gpt.util.to_list(first)
        e = expr(second)
        return_list = False
    else:
        assert ac is False
        if not gpt.util.is_list_instance(first, gpt.expr):
            return first

        e = expr(first)
        dst = None

    t("apply matrix ops")
    # apply matrix_operators
    e = apply_type_right_to_left(e, gpt.matrix_operator)

    t("fast return")
    # fast return if already a lattice
    if dst is None:
        if e.is_single(gpt.lattice):
            ue, uf, v = e.get_single()
            if uf == factor_unary.NONE and ue == expr_unary.NONE:
                return v
        elif e.is_num():
            return e.get_num()

    t("prepare")
    if dst is None:
        lat = e.lattice()
        if lat is None:
            # cannot evaluate to a lattice object, leave expression unevaluated
            return first
        return_list = isinstance(lat, list)
        lat = gpt.util.to_list(lat)
        grid = lat[0].grid
        nlat = len(lat)

    # verbose output
    if verbose:
        gpt.message("eval: " + str(e))

    if verbose_performance:
        cgpt.timer_begin()

    if dst is not None:
        t("cgpt.eval")
        for i, dst_i in enumerate(dst):
            dst_i.update(cgpt.eval(dst_i.v_obj, e.val, e.unary, ac, i))
        ret = dst
    else:
        assert ac is False
        t("get otype")
        # now find return type
        otype = get_otype_from_expression(e)

        ret = []

        for idx in range(nlat):
            t("cgpt.eval")
            res = cgpt.eval(None, e.val, e.unary, False, idx)
            t_obj, s_ot = (
                [x[0] for x in res],
                [x[1] for x in res],
            )

            assert s_ot == otype.v_otype

            t("lattice")
            ret.append(gpt.lattice(grid, otype, t_obj))

    t()
    if verbose_performance:
        t_cgpt = gpt.timer("cgpt_eval", True)
        t_cgpt += cgpt.timer_end()
        gpt.message(t)
        gpt.message(t_cgpt)

    if not return_list:
        return gpt.util.from_list(ret)

    return ret
