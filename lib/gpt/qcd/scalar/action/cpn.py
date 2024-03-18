#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2023  Mattia Bruno, Gabriele Morandi
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

# CP^{N-1} model

# S[z,l] = -N * beta * sum_n,mu (z_{n+mu}^dag * z_n * l_{n,mu} + z_n^dag * z_{n+mu} * l_{n,mu}^dag)
#        = -2 * N * beta * sum_n,mu [Re(z_{n+mu}^dag * z_n * l_{n,mu}) - 1]
class cpn(differentiable_functional):
    def __init__(self, N, b):
        self.N = N
        self.beta = b
        self.__name__ = f"cpn(N = {self.N}, beta = {self.beta})"

    # z = fields[0], l = fields[1:]
    def split(self, fields):
        return fields[0], fields[1:]

    def __call__(self, fields):
        z, l = self.split(fields)

        J = g.lattice(z)
        J[:] = 0.0
        for mu in range(z.grid.nd):
            J += g.cshift(z, mu, +1) * g.adj(l[mu])

        action = -2 * self.N * self.beta * (g.inner_product(J, z).real - z.grid.fsites * z.grid.nd)
        return action

    @differentiable_functional.multi_field_gradient
    def gradient(self, fields, dfields):
        def gradient_l(z, l, mu):
            frc = g.lattice(l[0])
            frc @= (
                2
                * self.beta
                * self.N
                * g.component.imag(g.trace(z * g.adj(g.cshift(z, mu, +1))) * l[mu])
            )
            frc.otype = l[0].otype.cartesian()
            return frc

        def gradient_z(z, l):
            J = g.lattice(z)
            J[:] = 0.0
            for mu in range(z.grid.nd):
                J += g.cshift(z, mu, +1) * g.adj(l[mu])
                J += g.cshift(z * l[mu], mu, -1)

            frc = g.lattice(z)
            frc @= -2 * self.beta * self.N * J

            frc -= g.trace(frc * g.adj(z)) * z
            return frc

        z, l = self.split(fields)
        frc = []
        for df in g.core.util.to_list(dfields):
            k = fields.index(df)
            if k == 0:
                frc.append(gradient_z(z, l))
            else:
                frc.append(gradient_l(z, l, k - 1))
        return frc

    # https://arxiv.org/abs/1102.1852
    def constrained_leap_frog(self, eps, z, mom_z):
        # TO DO: replace with g.adj(v1) * v2
        def dot(v1, v2):
            return g.trace(v2 * g.adj(v1))

        n = g.real(z.grid)
        n @= g.component.sqrt(g.component.real(dot(mom_z, mom_z)))

        # z'      =  cos(alpha) z + (1/|pi|) sin(alpha) mom_z
        # mom_z'  = -|pi| sin(alpha) z + cos(alpha) mom_z
        # alpha = eps |pi|
        _z = g.lattice(z)
        _z @= z

        cos = g.real(z.grid)
        cos @= g.component.cos(eps * n)

        sin = g.real(z.grid)
        sin @= g.component.sin(eps * n)

        z @= cos * _z + g(g.component.inv(n) * sin) * mom_z
        mom_z @= -g(n * sin) * _z + cos * mom_z
        del _z, cos, sin, n

    # https://arxiv.org/abs/1102.1852
    def draw(self, field, rng, constraint=None):
        if constraint is None:
            z = field
            rng.element(z)
            n = g.component.real(g.trace(z * g.adj(z)))
            z @= z * g.component.inv(g.component.sqrt(n))
        else:
            mom_z = field
            z = constraint
            rng.normal_element(mom_z)
            # TO DO change to z * g(g.adj(z) * mom_z)
            mom_z @= mom_z - g(z * g.adj(z)) * mom_z
