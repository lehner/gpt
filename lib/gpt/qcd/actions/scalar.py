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
import numpy
import gpt

def kappa2mass(k,l,D):
    return numpy.sqrt((1 - 2.0 * l)/k - 2.*D)

# -2 * kappa * sum_x,mu phi(x)^dag * phi(x+mu) + \sum_x |phi(x)|^2 + lambda * sum_x (|phi(x)|^2-1)^2
# sum_x (|phi(x)|^2-1)^2 = sum_x |phi(x)|^4 - 2 sum_x |phi(x)|^2 + vol
class phi4:
    def __init__(self, phi, m, l):
        self.phi = phi
        self.grid = phi.grid
        self.Nd = self.grid.nd
        self.m = m
        self.l = l
        self.kappa = (1 - 2.0 * l) / (2 * self.Nd + m ** 2)
        self.J = gpt.lattice(self.phi)

    def __str__(self):
        out = f'Scalar action phi^4\n'
        out += f' mass = {self.m}\n kappa = {self.kappa}\n lambda = {self.l}'
        return out
    
    def __call__(self):
        self.J[:] = 0
        for mu in range(self.Nd):
            self.J += gpt.cshift(self.phi, mu, 1)
        act = -2.0 * self.kappa * gpt.inner_product(self.J, gpt.adj(self.phi)).real

        p2 = gpt.norm2(self.phi)
        act += p2
        
        if (self.l != 0.0):
            p4 = gpt.norm2(self.phi * gpt.adj(self.phi))
            act += self.l * (p4 - 2.0 * p2 + self.grid.fsites)

        return act

    def setup_force(self):
        self.J[:] = 0
        for mu in range(self.Nd):
            self.J += gpt.cshift(self.phi, mu, 1)
            self.J += gpt.cshift(self.phi, mu, -1)

    def force(self, field):
        if field.v_obj != self.phi.v_obj:
            return None

        frc = gpt.lattice(self.phi)
        frc @= -2.0 * self.kappa * self.J
        frc += 2.0 * self.phi
        if self.l != 0.0:
#             p2 = gpt.norm2(self.phi)
            frc += 4.0 * self.l * self.phi * gpt.adj(self.phi) * self.phi
            frc -= 4.0 * self.l * self.phi
        return frc

# g sum_x, mu rho(x) phi(x)^dag phi(x+mu)
class rho_phi2:
    def __init__(self, phi, rho, g):
        self.g = g
        self.phi = phi
        self.grid = phi.grid
        self.Nd = self.grid.nd
        self.rho = rho
        self.J = gpt.lattice(self.phi)
        self.J2 = gpt.lattice(self.phi)
        
    def __call__(self):
        self.J[:] = 0
        for mu in range(self.Nd):
            self.J += gpt.cshift(self.phi, mu, 1)
            
        act = self.g * gpt.inner_product(self.J, self.phi * self.rho).real

        return act
    
    def setup_force(self):
        self.J[:] = 0
        self.J2[:] = 0
        for mu in range(self.Nd):
            self.J += gpt.cshift(self.phi, mu, 1)
            self.J2 += gpt.cshift(self.phi * self.rho, mu, -1)
        self.J2 += self.rho * self.J
        
    def force(self, field):
        frc = gpt.lattice(self.phi)

        if field.v_obj == self.phi.v_obj:
            frc @= self.g * self.J2
        elif field.v_obj == self.rho.v_obj:
            frc @= self.g * self.phi * self.J
        else:
            print(field.v_obj, self.rho.v_obj)
            raise Exception
        return frc