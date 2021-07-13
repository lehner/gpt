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

__all__ = ['quadratic_MDagM']

class quadratic_base(differentiable_functional):
    def __init__(self, M, inverter):
        self.M = M
        self.inverter = inverter

    def _updated(self, fields):
        U = fields[0:-1]
        psi = fields[-1]
        M = self.M.updated(U)
        return [M, U, psi]
    
def MMdag(M):
    def operator(dst, src):
        dst @= M * g(M.adj() * src)
    return operator

# S = phi^dag Mdag^-1 M^-1 phi = phi^dag (M Mdag)^-1 phi = (psi, psi)
# chi = Mdag^-1 psi = (M Mdag)^-1 phi
# psi = M^-1 phi    = Mdag chi
# dS = -(phi, (M Mdag)^-1 (dM Mdag + M dMdag) (M Mdag)^-1 phi)
#    = -(psi, (M^-1 dM + dMdag Mdag^-1) psi)
#    = - (chi, dM psi) - (psi, dMdag chi)
class quadratic_MDagM(quadratic_base):
    
    def __call__(self, fields):
        M, U, phi = self._updated(fields)
        
        chi = g.lattice(phi)
        chi @= self.inverter(MMdag(M)) * phi
        return g.norm2(g(M.adj() * chi))
    
        
    def draw(self, fields, rng):
        M, U, phi = self._updated(fields)
                
        eta = g.lattice(phi)
        rng.normal(eta, sigma=2.**-0.5) # 1/sqrt(2)

        phi @= M * eta
        return g.norm2(eta)

    
    def gradient(self, fields, dfields):
        M, U, phi = self._updated(fields)
        
        chi = g.lattice(phi)
        chi @= self.inverter(MMdag(M)) * phi
        psi = g.lattice(phi)
        psi @= M.adj() * chi
        
        frc = M.gradient(chi, psi)
        tmp = M.gradientDag(psi, chi)
        for mu in range(len(frc)):
            frc[mu] @= -frc[mu]-tmp[mu]
        del tmp
        
        dS = []
        for f in dfields:
            mu = fields.index(f)
            if mu<len(fields)-1:
                dS.append( frc[mu] )
            else:
                raise Expcetion('not implemented')
        return dS