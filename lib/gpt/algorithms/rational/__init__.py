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

import numpy
import gpt
from gpt.algorithms.rational.zolotarev import zolotarev_inverse_square_root

# Given the ratio of polynomials \prod_i (z+u_i)/(z+v_i)
# it returns the residues of the corresponding partial fractions
# decomposition
# \prod_i (z+u_i)/(z+v_i) = 1 + \sum_i r[i]/(z+v_i)
def partial_fractions(u,v):
    _v = numpy.array(v)
    n = len(u)
    r = numpy.zeros((n,))
    for i in range(n):
        _v[i] = 0.0
        z = -v[i]
        r[i] = numpy.prod(z + u) / numpy.prod(z + _v) * z
        _v[i] = v[i]
    return r

# \prod_i (z+u_i)/(z+v_i)
class rational_polynomial:
    def __init__(self, u, v, inverter=None):
        self.npoles = len(u)
        self.poles = numpy.array(v)
        self.r = partial_fractions(u,v)
        self.inverter = inverter
        
    def eval(self, x):
        f = 1.0
        for i, r in enumerate(self.r):
            f += r / (x + self.poles[i])
        return f
    
    def __str__(self):
        out =  f"Rational polynomial of degree {self.npoles}\n"
        out += "1\n"
        for i, r in enumerate(self.r):
            out += f"+ {r:g} / (x*x + {self.poles[i]:g}) + \n"
        return out
    
    def __call__(self, mat):
        if isinstance(mat, (float, complex, int, numpy.float32, numpy.float64)):
            return self.eval(mat)
        else:
            otype, grid, cb = None, None, None
            if type(mat) == gpt.matrix_operator:
                otype, grid, cb = mat.otype, mat.grid, mat.cb
                mat = mat.mat  # unwrap for performance benefit
            if self.inverter is None:
                raise NotImplementedError()
            
            mat_inv = self.inverter(mat, self.poles)
            
            def operator(dst, src):
                chi = [gpt.lattice(src) for _ in range(self.npoles)]
                mat_inv(chi, src)
                dst @= src
                for i in range(self.npoles):
                    dst += self.r[i] * chi[i]
                
            return gpt.matrix_operator(operator, grid=grid, otype=otype, cb=cb)

    def partial_fractions(self, mat):
        mat_inv = inverter(mat, self.poles)

        def operator(src):
            chi = [gpt.lattice(src) for _ in range(self.npoles)]
            mat_inv(chi, src)
            for i in range(self.npoles):
                chi[i] @= self.r[i] * chi[i]
            return chi
        
        return operator
        