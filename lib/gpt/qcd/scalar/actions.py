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

import gpt


class phi4:
    def __init__(self, phi, m, l):
        self.phi = phi
        self.grid = phi.grid
        self.Nd = self.grid.nd
        self.m = m
        self.l = l
        self.kappa = (1 - 2.0 * l) / (2 * self.Nd + m ** 2)
        self.J = gpt.lattice(self.phi)

    def __call__(self):
        self.J[:] = 0
        for mu in range(self.Nd):
            self.J += gpt.cshift(self.phi, mu, 1)
        act = -2.0 * self.kappa * gpt.inner_product(self.J, gpt.adj(self.phi)).real

        p2 = gpt.norm2(self.phi)
        act += p2 + self.l * (p2 - 1.0) ** 2

        return act

    def setup_force(self):
        self.J[:] = 0
        for mu in range(self.Nd):
            self.J += gpt.cshift(self.phi, mu, 1)
            self.J += gpt.cshift(self.phi, mu, -1)

    def force(self, field):
        if field.v_obj != self.phi.v_obj:
            raise Exception

        frc = gpt.lattice(self.phi)
        frc @= -2.0 * self.kappa * self.J
        frc += 2.0 * self.phi
        if self.l != 0.0:
            frc += 4.0 * self.l * gpt.adj(self.phi) * self.phi * self.phi
            frc += 4.0 * self.l * self.phi
        frc[:].imag = 0
        return frc
