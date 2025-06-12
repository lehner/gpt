#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2024  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.core.group import differentiable_functional


class polar_regulator(differentiable_functional):
    def __init__(self, lam, kap, c):
        self.lam = lam
        self.kap = kap
        self.c = c

    def __call__(self, fields):
        I = g.identity(fields[0])
        Nc = fields[0].otype.Ndim
        r = 0.0
        for mu in range(len(fields)):
            r += (self.lam / 2 / Nc) * g.sum(g.component.pow(self.c)(g.trace((fields[mu] - I) * (fields[mu] - I)))).real
            r -= (self.kap / Nc) * g.sum(g.component.log(g.matrix.det(fields[mu]))).real
        return r

    def gradient(self, fields, dfields):
        # log(det(A + dA)) = log(det(A(1+inv(A)dA))) = log(det(A)) + log(1+tr(invA dA))
        #                  = log(det(A)) + tr(invA dA)
        dAdS = []
        I = g.identity(dfields[0])
        Nc = fields[0].otype.Ndim
        for df in dfields:
            x = g(2.0 * (self.lam / 2 / Nc) * self.c * g.component.pow(self.c - 1)(g.trace((df - I) * (df - I))) * (df - I) - (self.kap / Nc) * g.matrix.inv(df))
            dAdS.append(x)
        return dAdS
