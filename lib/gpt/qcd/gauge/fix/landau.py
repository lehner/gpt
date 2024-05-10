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
import gpt as g
import numpy as np
from gpt.core.group import differentiable_functional


class landau(differentiable_functional):
    def __init__(self, U):
        self.U = U

    def __call__(self, V):
        V = g.util.from_list(V)
        return sum([g.sum(g.trace(u)) for u in g.qcd.gauge.transformed(self.U, V)]).real * (-2.0)

    @differentiable_functional.single_field_gradient
    def gradient(self, V):
        A = [
            g(g.qcd.gauge.project.traceless_anti_hermitian(u) / 1j)
            for u in g.qcd.gauge.transformed(self.U, V)
        ]
        dmuAmu = V.new()
        dmuAmu.otype = V.otype.cartesian()
        dmuAmu = g(0.0 * dmuAmu)
        for mu, Amu in enumerate(A):
            dmuAmu += Amu - g.cshift(Amu, mu, -1)
        return dmuAmu
