#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020-22  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt.default as default
import gpt as g
import numpy as np


c = {}


def cayley_hamilton_function_and_gradient_3(iQ, gradient_prime, c):
    # For now use Cayley Hamilton Decomposition for traceless Hermitian 3x3 matrices,
    # see https://arxiv.org/pdf/hep-lat/0311018.pdf

    I = g.identity(iQ)

    Q = g(-1j * iQ)
    Q2 = g(Q * Q)
    Q3 = g(Q * Q2)

    c0 = g(g.trace(Q3) * (1.0 / 3.0))
    c1 = g(g.trace(Q2) * (1.0 / 2.0))

    one = g.identity(c0)

    c0max = g(2.0 * g.component.pow(1.5)(c1 / 3.0))

    theta = g.component.acos(c0 * g.component.inv(c0max))
    u = g(g.component.sqrt(c1 / 3.0) * g.component.cos(theta / 3.0))
    w = g(g.component.sqrt(c1) * g.component.sin(theta / 3.0))
    u2 = g(u * u)
    w2 = g(w * w)
    fden = g.component.inv(9.0 * u2 - w2)
    fden2 = g(fden * fden / 2.0)

    xi0 = g(g.component.sin(w) * g.component.inv(w))
    xi1 = g(g.component.cos(w) * g.component.inv(w2) - g.component.sin(w) * g.component.inv(w * w2))
    cosw = g.component.cos(w)

    emiu = g(g.component.cos(u) - 1j * g.component.sin(u))
    e2iu = g(g.component.cos(2.0 * u) + 1j * g.component.sin(2.0 * u))

    # can do in stencil:
    with c.code() as cc:
        ixi0 = cc(1j * xi0)
        h0 = cc(e2iu * (u2 - w2) + emiu * ((8.0 * u2 * cosw) + (2.0 * u * (3.0 * u2 + w2) * ixi0)))
        h1 = cc(e2iu * (2.0 * u) - emiu * ((2.0 * u * cosw) - (3.0 * u2 - w2) * ixi0))
        h2 = cc(e2iu - emiu * (cosw + (3.0 * u) * ixi0))

        f0 = cc(h0 * fden)
        f1 = cc(h1 * fden)
        f2 = cc(h2 * fden)

        r01 = cc(
            (2.0 * u + 1j * 2.0 * (u2 - w2)) * e2iu
            + emiu
            * (
                (16.0 * u * cosw + 2.0 * u * (3.0 * u2 + w2) * xi0)
                + 1j * (-8.0 * u2 * cosw + 2.0 * (9.0 * u2 + w2) * xi0)
            )
        )

        r11 = cc(
            (2.0 * one + 4j * u) * e2iu
            + emiu
            * ((-2.0 * cosw + (3.0 * u2 - w2) * xi0) + 1j * ((2.0 * u * cosw + 6.0 * u * xi0)))
        )

        r21 = cc(2j * e2iu + emiu * (-3.0 * u * xi0 + 1j * (cosw - 3.0 * xi0)))

        r02 = cc(
            -2.0 * e2iu + emiu * (-8.0 * u2 * xi0 + 1j * (2.0 * u * (cosw + xi0 + 3.0 * u2 * xi1)))
        )

        r12 = cc(emiu * (2.0 * u * xi0 + 1j * (-cosw - xi0 + 3.0 * u2 * xi1)))

        r22 = cc(emiu * (xi0 - 1j * (3.0 * u * xi1)))

        b10 = cc(2.0 * u * r01 + (3.0 * u2 - w2) * r02 - (30.0 * u2 + 2.0 * w2) * f0)
        b11 = cc(2.0 * u * r11 + (3.0 * u2 - w2) * r12 - (30.0 * u2 + 2.0 * w2) * f1)
        b12 = cc(2.0 * u * r21 + (3.0 * u2 - w2) * r22 - (30.0 * u2 + 2.0 * w2) * f2)

        b20 = cc(r01 - (3.0 * u) * r02 - (24.0 * u) * f0)
        b21 = cc(r11 - (3.0 * u) * r12 - (24.0 * u) * f1)
        b22 = cc(r21 - (3.0 * u) * r22 - (24.0 * u) * f2)

        b10 *= fden2
        b11 *= fden2
        b12 *= fden2
        b20 *= fden2
        b21 *= fden2
        b22 *= fden2

    c.execute()

    # assemble results
    B1 = g(b10 * I + b11 * Q + b12 * Q2)
    B2 = g(b20 * I + b21 * Q + b22 * Q2)

    U_Sigma_prime = gradient_prime

    exp_iQ = g(f0 * I + f1 * Q + f2 * Q2)

    Gamma = g(
        g.trace(U_Sigma_prime * B1) * Q
        + g.trace(U_Sigma_prime * B2) * Q2
        + f1 * U_Sigma_prime
        + f2 * Q * U_Sigma_prime
        + f2 * U_Sigma_prime * Q
    )

    Lambda = g.qcd.gauge.project.traceless_hermitian(Gamma)

    return exp_iQ, Lambda


def cayley_hamilton_function_and_gradient(x, dx, c):
    if x.otype.shape[0] == 3:
        return cayley_hamilton_function_and_gradient_3(x, dx, c)

    raise NotImplementedError()


default_exp_cache = {}


def function(i, cache=default_exp_cache):
    i = g.eval(i)  # accept expressions
    if i.grid.precision != g.double:
        x = g.convert(i, g.double)
    else:
        x = g.copy(i)

    n = g.object_rank_norm2(x) ** 0.5 / x.grid.gsites * x.grid.Nprocessors
    maxn = 0.01
    ns = 0
    if n > maxn:
        ns = int(np.log2(n / maxn))
        x /= 2**ns

    o = g.identity(x)
    xn = g.copy(x)

    if isinstance(x, g.lattice) and len(x.v_obj) == 1:
        tag = f"{x.otype.__name__}_{x.grid}"

        if tag not in cache:
            points = [(0,) * x.grid.nd]
            _o = 0
            _xn = 1
            _x = 2
            nfac = 1.0
            code = [(_o, _o, 1.0, [(_xn, 0, 0)])]
            order = 19
            for j in range(2, order + 1):
                nfac /= j
                code.append((_xn, -1, 1.0, [(_xn, 0, 0), (_x, 0, 0)]))
                code.append((_o, _o, nfac, [(_xn, 0, 0)]))

            cache[tag] = g.local_stencil.matrix(x, points, code)

        cache[tag](o, xn, x)
    else:
        o += xn
        nfac = 1.0
        order = 19
        for j in range(2, order + 1):
            nfac /= j
            xn @= xn * x
            o += xn * nfac

    for j in range(ns):
        o @= o * o
    if i.grid.precision != g.double:
        r = g.lattice(i)
        g.convert(r, o)
        o = r

    return o


def function_and_gradient(x, dx):
    global c

    if x.grid.precision != g.double:
        x_dp = g.convert(x, g.double)
        dx_dp = g.convert(dx, g.double)
        A, B = function_and_gradient(x_dp, dx_dp)
        return g.convert(A, x.grid.precision), g.convert(B, x.grid.precision)

    key = f"{x.otype.__name__};{x.grid}"
    if key not in c:
        c[key] = g.compiler()
    return cayley_hamilton_function_and_gradient(x, dx, c[key])
