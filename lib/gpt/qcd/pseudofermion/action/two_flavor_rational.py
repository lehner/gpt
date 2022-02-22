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

import gpt as g
from gpt.qcd.pseudofermion.action.base import action_base
from gpt.qcd.pseudofermion.action.schur_differentiable_operator import *

#
#     [  (A+v_0) (A+v_1) (A+v_2) ... ]
# det [  ------- ------- -------     ]     u_i, v_i reals
#     [  (A+u_0) (A+u_1) (A+u_2) ... ]
#
#             [  (A+u_0) (A+u_1) (A+u_2) ... ]
# S = phi^dag [  ------- ------- -------     ] phi
#             [  (A+v_0) (A+v_1) (A+v_2) ... ]
#
#             [        r_0       r_1         ]
#   = phi^dag [1  +  ------- + ------- + ... ] phi
#             [      (A+v_0)   (A+v_1)   ... ]
#  
#   = phi^dag phi + \sum_i phi^dag chi_i ;   chi_i = r_i [A+v_i]^{-1} phi 
#
# dS = phi^dag [\sum_i r_i [A + v_i]^{-1} dA [A + v_i]^{-1} ] phi
#    = \sum_i r_i chi_i^dag (dM Mdag + M dMdag) chi_i
#

class two_flavor_rational_base(action_base):
    def __init__(self, M, rational, operator):
        self.rational = rational
        super().__init__(M, None, operator)
        
    def Mg5s(self, M, src, s):
        return g.eval(M * g.gamma[5] * src + 1j * s * src)
    
    def draw(self, fields, rng):
        pf = self.rational.partial_fractions(self.operator.MMdag(M))
        chi = pf(eta)
        phi @= eta
        for i, c in enumerate(chi):
            phi += self.Mg5s(self.operator.M(M), c, self.rational.poles[i]**0.5) 
        return g.norm2(eta)
    
    def __call__(self, fields):
        M, U, phi = self._updated(fields)
        
        rat = self.rational(self.operator.MMdag(M))
        
        chi = g.lattice(phi)
        chi @= rat * phi
        return g.inner_product(chi, phi).real
    
    def gradient(self, fields, dfields):
        M, U, phi = self._updated(fields)

        pf = self.rational.partial_fractions(self.operator.MMdag(M))
        chi = pf(phi)

        frc = self._allocate_force(U)
        psi = g.lattice(phi)
        for _chi in chi:
            psi @= self.operator.Mdag(M) * _chi;
            self._accumulate(frc, self.operator.Mderiv(M)(_chi, psi), -1)
            self._accumulate(frc, self.operator.MderivDag(M)(psi, _chi), -1)

        dS = []
        for f in dfields:
            mu = fields.index(f)
            if mu < len(fields) - 1:
                dS.append(g.qcd.gauge.project.traceless_hermitian(frc[mu]))
            else:
                raise Expcetion("not implemented")
        return dS
        

# A = M Mdag
class two_flavor_rational(two_flavor_rational_base):
    def __init__(self, M, rational):
        super().__init__(M, rational, MMdag)
        
# A = Mhat Mhatdag
class two_flavor_rational_evenodd(two_flavor_rational_base):
    def __init__(self, M, rational):
        super().__init__(M, rational, MMdag_evenodd)
    