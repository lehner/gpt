#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import numpy as np


# returns polynomial coefficients up to order n
# assuming dst was obtained by applying a Krylov method in w to src
def krylov(dst, w, src, n):
    # normalize src
    alpha = 1.0 / g.norm2(src) ** 0.5
    src = g(alpha * src)
    dst = g(alpha * dst)

    # numerically stable krylov space
    k = [src]
    V = np.zeros(shape=(n + 1, n + 1), dtype=np.complex128)
    V[0, 0] = 1
    for i in range(n):
        ktop = g(w * k[-1])
        Vline = g.inner_product(k, ktop)[:, 0]
        ksub = g.lattice(ktop)
        g.linear_combination(ksub, k, Vline)
        ktop -= ksub
        alpha = g.norm2(ktop) ** 0.5
        ktop /= alpha
        V[i + 1, i + 1] = 1 / alpha
        V[i + 1, 0 : i + 1] = Vline / alpha
        k.append(ktop)

    # get matrix elements (should be able to construct this out of V,
    # for purpose of this function, overhead is OK)
    # f = <k_j| f | k_0> = <k_j| sum_i c_i w^i | k_0> = <k_j| w^i | k_0> c_i
    M = np.zeros(shape=(n + 1, n + 1), dtype=np.complex128)
    t = g.copy(k[0])
    for i in range(n + 1):
        vv = g.inner_product(k, t)[:, 0]
        M[:, i] = vv
        t = g(w * t)

    f = g.inner_product(k, dst)
    return (np.linalg.inv(M) @ f)[:, 0]
