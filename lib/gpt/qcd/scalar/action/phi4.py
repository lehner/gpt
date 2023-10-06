#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Mattia Bruno
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
import numpy


# -2 * kappa * sum_x,mu phi(x)^dag * phi(x+mu) + \sum_x |phi(x)|^2 + lambda * sum_x (|phi(x)|^2-1)^2
# sum_x (|phi(x)|^2-1)^2 = sum_x |phi(x)|^4 - 2 sum_x |phi(x)|^2 + vol
class phi4(differentiable_functional):
    def __init__(self, k, l):
        self.l = l
        self.kappa = k
        self.__name__ = f"phi4({self.kappa},{self.l})"

    def kappa_to_mass(self, k, l, D):
        return numpy.sqrt((1 - 2.0 * l) / k - 2.0 * D)

    def __call__(self, phi):
        J = None
        act = 0.0
        for p in g.core.util.to_list(phi):
            if J is None:
                J = g.lattice(p)

            J[:] = 0
            for mu in range(p.grid.nd):
                J += g.cshift(p, mu, 1)
            act += -2.0 * self.kappa * g.inner_product(J, g.adj(p)).real

            p2 = g.norm2(p)
            act += p2

            if self.l != 0.0:
                p4 = g.norm2(p * g.adj(p))
                act += self.l * (p4 - 2.0 * p2 + p.grid.fsites)

        return act

    @differentiable_functional.single_field_gradient
    def gradient(self, phi):
        J = g.lattice(phi)
        frc = g.lattice(phi)

        J[:] = 0
        for mu in range(phi.grid.nd):
            J += g.cshift(phi, mu, +1)
            J += g.cshift(phi, mu, -1)

        frc @= -2.0 * self.kappa * J
        frc += 2.0 * phi
        if self.l != 0.0:
            frc += 4.0 * self.l * phi * g.adj(phi) * phi
            frc -= 4.0 * self.l * phi

        return frc
