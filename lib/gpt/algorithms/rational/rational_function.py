#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2022  Mattia Bruno
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

# for more details see documentation/algorithms/rational.ipynb

import numpy
import gpt as g


# residues of the partial fractions
# \prod_i (z-u_i)/(z-v_i) = 1 + \sum_i r[i]/(z-v_i)
def partial_fractions(u, v):
    _v = numpy.array(v)
    n = len(v)
    r = numpy.zeros((n,))
    for i in range(n):
        _v[i] = 0.0
        z = v[i]
        r[i] = numpy.prod(z - u) / numpy.prod(z - _v) * z
        _v[i] = v[i]
    return r


# \prod_i (z-u_i)/(z-v_i)
class rational_function:
    def __init__(self, zeros, poles, norm=1.0, inverter=None):
        self.npoles = len(poles)
        self.poles = numpy.array(poles)
        self.zeros = numpy.array(zeros)
        self.norm = norm
        self.r = partial_fractions(self.zeros, self.poles)
        self.inverter = inverter
        if len(poles) == len(zeros):
            self.pf0 = 1.0
        elif len(poles) > len(zeros):
            self.pf0 = 0.0
        else:
            raise Exception("Rational function ill behaved at infinity")

    def eval(self, x):
        f = self.pf0
        for i, r in enumerate(self.r):
            f += r / (x - self.poles[i])
        return f * self.norm

    def __str__(self):
        out = f"Rational function of degree {self.npoles}\n"
        out += f"{self.norm:g}({self.pf0} + "
        for i, r in enumerate(self.r):
            out += f"\n+ {r:g} / (x - {self.poles[i]:g})"
        out += "\n)"
        return out

    def __call__(self, mat):
        if g.util.is_num(mat):
            return self.eval(mat)
        else:
            vector_space = None
            if isinstance(mat, g.matrix_operator):
                vector_space = mat.vector_space
                mat = mat.mat
                # remove wrapper for performance benefits

            if self.inverter is None:
                raise NotImplementedError()

            pf = self.partial_fractions(mat)

            def operator(dst, src):
                chi = pf(src)
                dst @= src
                if self.pf0 == 0.0:
                    dst[:] = 0
                for i, c in enumerate(chi):
                    dst += self.r[i] * c
                if self.norm != 1.0:
                    dst *= self.norm

            return g.matrix_operator(mat=operator, vector_space=vector_space)

    def inv(self):
        return rational_function(self.poles, self.zeros, 1.0 / self.norm, self.inverter)

    # chi_i = [A+v_i]^{-1} phi
    def partial_fractions(self, mat):
        self.inverter.shifts = -self.poles
        mat_inv = self.inverter(mat)

        def operator(src):
            chi = [g.lattice(src) for _ in range(self.npoles)]
            mat_inv(chi, src)
            return chi

        return operator
