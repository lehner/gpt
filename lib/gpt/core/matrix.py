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
import gpt, cgpt, numpy

# matrix exponential
def exp(i):
    t = gpt.timer("exp")
    t("eval")
    i = gpt.eval(i)  # accept expressions
    t("prep")
    if i.grid.precision != gpt.double:
        x = gpt.convert(i, gpt.double)
    else:
        x = gpt.copy(i)
    n = gpt.norm2(x) ** 0.5 / x.grid.gsites
    order = 19
    maxn = 0.01
    ns = 0
    if n > maxn:
        ns = int(numpy.log2(n / maxn))
        x /= 2 ** ns
    o = gpt.lattice(x)
    t("mem")
    o[:] = 0
    nfac = 1.0
    xn = gpt.copy(x)
    t("id")
    o @= gpt.identity(o)
    t("add")
    o += xn
    t("loop")
    for j in range(2, order + 1):
        nfac /= j
        xn @= xn * x
        o += xn * nfac
    t("reduce")
    for j in range(ns):
        o @= o * o
    t("conv")
    if i.grid.precision != gpt.double:
        r = gpt.lattice(i)
        gpt.convert(r, o)
        o = r
    t()
    # gpt.message(t)
    return o


def log(i, convergence_threshold=0.5):
    i = gpt.eval(i)
    # i = n*(1 + x), log(i) = log(n) + log(1+x)
    # x = i/n - 1, |x|^2 = <i/n - 1, i/n - 1> = |i|^2/n^2 + |1|^2 - (<i,1> + <1,i>)/n
    # d/dn |x|^2 = -2 |i|^2/n^3 + (<i,1> + <1,i>)/n^2 = 0 -> 2|i|^2 == n (<i,1> + <1,i>)
    if i.grid.precision != gpt.double:
        x = gpt.convert(i, gpt.double)
    else:
        x = gpt.copy(i)
    lI = gpt.identity(gpt.lattice(x))
    n = gpt.norm2(x) / gpt.inner_product(x, lI).real
    x /= n
    x -= lI
    n2 = gpt.norm2(x) ** 0.5 / x.grid.gsites
    order = 8 * int(16 / (-numpy.log10(n2)))
    assert n2 < convergence_threshold
    o = gpt.copy(x)
    xn = gpt.copy(x)
    for j in range(2, order + 1):
        xn @= xn * x
        o -= xn * ((-1.0) ** j / j)
    o += lI * numpy.log(n)
    if i.grid.precision != gpt.double:
        r = gpt.lattice(i)
        gpt.convert(r, o)
        o = r
    return o


def inv(A):
    A = gpt.eval(A)
    assert type(A) == gpt.lattice
    A_inv = gpt.lattice(A)
    to_list = gpt.util.to_list
    cgpt.invert_matrix(to_list(A_inv), to_list(A))
    return A_inv


def det(A):
    A = gpt.eval(A)
    assert type(A) == gpt.lattice
    r = gpt.complex(A.grid)
    to_list = gpt.util.to_list
    cgpt.determinant(r.v_obj[0], to_list(A))
    return r
