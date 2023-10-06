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
# Note: we use the proper order of the chebyshev_t
#       in contrast to current Grid
#
import gpt as g
import math
import numpy as np


def make_list(a):
    if isinstance(a, list):
        return a
    return [a]


def undo_list(a):
    if isinstance(a, list) and len(a) == 1:
        return a[0]
    return a


def coeffs_func(lo, hi, orders, funcs):
    ret = []
    for i in range(len(orders)):
        coeffs = []
        order = orders[i]
        func = funcs[i]
        for j in range(order):
            s = 0.0
            for k in range(order):
                y = math.cos(math.pi * (k + 0.5) / order)
                x = 0.5 * (y * (hi - lo) + (hi + lo))
                f = func(x)
                s += f * math.cos(j * math.pi * (k + 0.5) / order)
            coeffs.append(s * 2.0 / order)
        ret.append(coeffs)
    return ret


def coeffs_order(orders):
    return [[0.0] * (order - 1) + [1.0] for order in orders]


class chebyshev:
    """Chebyshev Polynomials

    Args:
        params (dict): A dictionary containing the in input parameters to be saved as attributes.

    """

    @g.params_convention(low=None, high=None, order=None, func=None)
    def __init__(self, params):
        self.params = params
        self.hi = params["high"]
        self.lo = params["low"]
        # order as well as func can be lists to compute multiple at once
        self.order = [y + 1 for y in make_list(params["order"])]
        self.morder = max(self.order)
        self.n = len(self.order)
        if params["func"] is not None:
            self.func = make_list(params["func"])
            assert self.n == len(self.func)
            self.coeffs = coeffs_func(self.lo, self.hi, self.order, self.func)
        else:
            self.coeffs = coeffs_order(self.order)

    def eval(self, x):
        """Summary of eval

        Some additonal explanation of the eval function

        Args:
            x (float): Input x

        Returns:
            Some return value which was a list
        """
        y = (x - 0.5 * (self.hi + self.lo)) / (0.5 * (self.hi - self.lo))
        T0 = 1
        T1 = y
        s = [0.5 * c[0] * T0 for c in self.coeffs]
        for i, c in enumerate(self.coeffs):
            s[i] += c[1] * T1

        Tn = T1
        Tnm = T0

        for i in range(2, self.morder):
            Tnp = 2 * y * Tn - Tnm
            Tnm = Tn
            Tn = Tnp
            for j in range(self.n):
                if len(self.coeffs[j]) > i:
                    s[j] += Tn * self.coeffs[j][i]

        return undo_list(s)

    def evalD(self, x):
        y = (x - 0.5 * (self.hi + self.lo)) / (0.5 * (self.hi - self.lo))
        U0 = 1
        U1 = 2 * y
        s = [c[1] * U0 for c in self.coeffs]
        for i, c in enumerate(self.coeffs):
            s[i] += c[2] * U1 * 2.0

        Un = U1
        Unm = U0

        for i in range(2, self.morder):
            Unp = 2 * y * Un - Unm
            Unm = Un
            Un = Unp
            for j in range(self.n):
                if len(self.coeffs[j]) > i + 1:
                    s[j] += Un * self.coeffs[j][i + 1] * (i + 1)

        return undo_list([v / (0.5 * (self.hi - self.lo)) for v in s])

    def __call__(self, mat):
        if isinstance(mat, (float, complex, int)):
            return self.eval(mat)
        else:
            vector_space = None
            if isinstance(mat, g.matrix_operator):
                vector_space = mat.vector_space
                mat = mat.mat  # unwrap for performance benefit

            def evalOp(dst, src):
                dst = make_list(dst)
                xscale = 2.0 / (self.hi - self.lo)
                mscale = -(self.hi + self.lo) / (self.hi - self.lo)
                T0, T1, T2, y = (
                    g.copy(src),
                    g.lattice(src),
                    g.lattice(src),
                    g.lattice(src),
                )
                Tnm, Tn, Tnp = T0, T1, T2
                mat(y, T0)
                T1 @= y * xscale + src * mscale
                for i in range(self.n):
                    dst[i] @= (0.5 * self.coeffs[i][0]) * T0 + self.coeffs[i][1] * T1
                for n in range(2, self.morder):
                    mat(y, Tn)
                    y @= xscale * y + mscale * Tn
                    Tnp @= 2.0 * y - Tnm
                    for i in range(self.n):
                        if len(self.coeffs[i]) > n:
                            dst[i] += self.coeffs[i][n] * Tnp
                    Tnm, Tn, Tnp = Tn, Tnp, Tnm

            return g.matrix_operator(evalOp, vector_space=vector_space)
