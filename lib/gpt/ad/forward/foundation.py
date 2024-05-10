#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt as g


python_sum = sum


def inner_product(sx, sy, use_accelerator):
    assert len(sx) == 1 and len(sy) == 1
    sx = sx[0]
    sy = sy[0]
    return {(0, 0): sx.distribute2(sy, lambda a, b: g.inner_product(a, b, use_accelerator))}


def rank_inner_product(sx, sy, use_accelerator):
    assert len(sx) == 1 and len(sy) == 1
    sx = sx[0]
    sy = sy[0]
    return {(0, 0): sx.distribute2(sy, lambda a, b: g.rank_inner_product(a, b, use_accelerator))}


def norm2(sx):
    assert len(sx) == 1
    return [inner_product(sx, sx, True)[0, 0]]


def object_rank_norm2(sx):
    assert len(sx) == 1
    return python_sum([g.object_rank_norm2(sx[0].terms[t]) for t in sx[0].terms])


def cshift(sx, mu, disp, none=None):
    assert none is None
    return sx.distribute1(lambda a: g.cshift(a, mu, disp))


def trace(sx, t):
    return sx.distribute1(lambda a: g.trace(a, t))


def adj(sx):
    return sx.distribute1(lambda a: g.adj(a))


def sum(sx):
    return sx.distribute1(lambda a: g.sum(a))


def identity(sx):
    idsx = None
    for t in sx.terms:
        idsx = g.identity(sx[t])
        break
    assert idsx is not None
    return g.ad.forward.series({g.ad.forward.infinitesimal({}): idsx}, sx.landau_O)


def infinitesimal_to_cartesian(src, dsrc):
    return dsrc[1].otype.infinitesimal_to_cartesian(src, dsrc)


def group_inner_product(left, right):
    return left.distribute2(right, lambda a, b: g.group.inner_product(a, b))


def copy(dst, src):
    for i in range(len(dst)):
        dst[i] @= src[i]


def convert(first, second):
    if isinstance(second, g.ot_base) and first.otype.__name__ != second.__name__:
        assert second.__name__ in first.otype.ctab
        tmp = g.ad.forward.series(
            {t: g.lattice(first.grid, second) for t in first.terms}, first.landau_O
        )
        first.otype.ctab[second.__name__](tmp, first)
        tmp.otype = second
        return tmp

    raise Exception(f"Not yet implemented for {type(first)} x {type(second)}")


def matrix_det(sx):
    def df(x, dx, maxn):
        # det(sx + dsx) = det(sx(1 + sx^-1 dsx))
        #               = det(sx) det(1 + sx^-1 dsx)
        #               = det(sx) (1 + tr(sx^-1 dsx) + O(dsx^2))
        # higher-order:
        # det(A) = exp ln det(A) = exp tr ln A
        # det(sx + dsx) = exp tr ln (sx + dsx)
        # ln(sx + dsx) = ln(sx) + ln(1 + sx^-1 dsx)    | correct under exp tr
        #              = ln(sx) + sx^-1 dsx - (1/2) sx^-1 dsx sx^-1 dsx + O(dsx^2)
        # tr[...]      = tr[ln(sx)] + tr[sx^-1 dsx] - 1/2 tr[sx^-1 dsx sx^-1 dsx] + ...
        # exp tr[...]  = det(sx) * exp(tr[sx^-1 dsx]) * exp(- 1/2 tr[sx^-1 dsx sx^-1 dsx]) * ...
        # exp tr[...]  = det(sx) * (1 + tr[sx^-1 dsx] + 1/2 * tr[sx^-1 dsx]^2) * (1 - 1/2 tr[sx^-1 dsx sx^-1 dsx])
        # det(sx + dsx)= det(sx) * (1 + tr[sx^-1 dsx] + 1/2 * tr[sx^-1 dsx]^2) * (1 - 1/2 tr[sx^-1 dsx sx^-1 dsx])
        v0 = g.ad.forward.series(g.matrix.det(x), dx.landau_O)
        if maxn >= 0:
            v = v0
        if maxn >= 2:
            adjx = dx * g.matrix.inv(x)
            tr_adjx = g.trace(adjx)
            v += v0 * tr_adjx
        if maxn >= 3:
            adjx2 = adjx * adjx
            tr_adjx2 = g.trace(adjx2)
            v += v0 * (tr_adjx * tr_adjx - tr_adjx2) / 2.0
        if maxn >= 4:
            raise Exception(f"{maxn-1}-derivative of g.matrix.det not yet implemented")
        v.otype = v0.otype
        return v

    return sx.function(df)


def component_simple_map(operator, numpy_operator, extra_params, first, second):
    if operator == "pow":
        assert second is None
        exponent = extra_params["exponent"]
        pow = g.component.pow

        def df(x, dx, maxn):
            # (x + dx)**exponent = x**exponent + exponent * x**(exponent-1) dx
            #                      + 1/2 * exponent * (exponent-1) * x**(exponent-2) dx**2
            v = g.ad.forward.series(pow(exponent)(x), dx.landau_O)
            fac = None
            for i in range(1, maxn):
                fac = ((exponent + 1 - i) / i) * (g.component.multiply(dx, fac) if i > 1 else dx)
                v += g.component.multiply(
                    fac, g.ad.forward.series(pow(exponent - i)(x), dx.landau_O)
                )
            v.otype = x.otype
            return v

        return first.function(df)
    raise Exception(f"component-wise operator {operator} not implemented in forward-AD")


def component_multiply(sx, sy):
    return sx.distribute2(sy, lambda a, b: g.component.multiply(a, b))
