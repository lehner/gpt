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
from gpt.qcd.pseudofermion.action.base import action_base
from gpt.qcd.pseudofermion.action.schur_differentiable_operator import *


# S = phi^dag Mdag^-1 M^-1 phi = phi^dag (M Mdag)^-1 phi = (psi, psi)
# chi = Mdag^-1 psi = (M Mdag)^-1 phi
# psi = M^-1 phi    = Mdag chi
# dS = - (phi, (M Mdag)^-1 (dM Mdag + M dMdag) (M Mdag)^-1 phi)
#    = - (chi, dM psi) - (psi, dMdag chi)
class two_flavor_base(action_base):
    def __call__(self, fields):
        M, U, phi = self._updated(fields)

        chi = g.lattice(phi)
        chi @= self.inverter(self.operator.MMdag(M)) * phi
        return g.inner_product(phi, chi).real

    def draw(self, fields, rng):
        M, U, phi = self._updated(fields)

        eta = g.lattice(phi)
        rng.cnormal(eta, sigma=2.0**-0.5)  # 1/sqrt(2)

        phi @= self.operator.M(M) * eta
        return g.norm2(eta)

    def gradient(self, fields, dfields):
        M, U, phi = self._updated(fields)

        chi = g.lattice(phi)
        chi @= self.inverter(self.operator.MMdag(M)) * phi
        psi = g.lattice(phi)
        psi @= self.operator.Mdag(M) * chi

        frc = self._allocate_force(U)
        self._accumulate(frc, self.operator.Mderiv(M)(chi, psi), -1)
        self._accumulate(frc, self.operator.MderivDag(M)(psi, chi), -1)

        dS = []
        for f in dfields:
            mu = fields.index(f)
            if mu < len(fields) - 1:
                dS.append(g.qcd.gauge.project.traceless_hermitian(frc[mu]))
            else:
                raise Exception("not implemented")
        return dS


class two_flavor(two_flavor_base):
    def __init__(self, M, inverter):
        super().__init__(M, inverter, MMdag)


#      ( EE EO )   ( 1    EO OO^-1 ) ( Mhat 0 ) ( 1  0  )
#  M = ( OE OO ) = ( 0    1        ) ( 0    1 ) ( OE OO )
# Mhat = EE - EO OO^-1 OE
# S = psi^dag (Mhat Mhatdag)^-1 psi


class two_flavor_evenodd_schur(two_flavor_base):
    def __init__(self, M, inverter):
        super().__init__(M, inverter, MMdag_evenodd())


class two_flavor_evenodd(differentiable_functional):
    def __init__(self, M, inverter):
        self.M = M
        self.inverter = inverter
        self.two_flavor_evenodd_schur = two_flavor_evenodd_schur(M, inverter)

    def _get_fields(self, fields):
        U = fields[0:-1]
        phi = fields[-1]
        phi_o = g.lattice(self.M.F_grid_eo, phi.otype)
        phi_e = g.lattice(self.M.F_grid_eo, phi.otype)
        g.pick_checkerboard(g.odd, phi_o, phi)
        g.pick_checkerboard(g.even, phi_e, phi)
        return [U, phi_e, phi_o]

    def __call__(self, fields):
        U, phi_e, phi_o = self._get_fields(fields)
        return self.two_flavor_evenodd_schur(U + [phi_o])  # + detDee

    def draw(self, fields, rng):
        U = fields[0:-1]
        phi = fields[-1]

        phi_o = g.lattice(self.M.F_grid_eo, phi.otype)
        g.pick_checkerboard(g.odd, phi_o, phi)
        act_o = self.two_flavor_evenodd_schur.draw(U + [phi_o], rng)

        phi[:] = 0
        g.set_checkerboard(phi, phi_o)
        return act_o

    def gradient(self, fields, dfields):
        U, phi_e, phi_o = self._get_fields(fields)
        count = len(fields) - 1
        for f in dfields:
            mu = fields.index(f)
            if mu < len(fields) - 1:
                count -= 1
        if count == 0:
            return self.two_flavor_evenodd_schur.gradient(U + [phi_o], dfields)
        else:
            raise Exception("not implemented")


# det (M1 M1dag/M2 M2dag) -> S = phi^dag M2dag (M1 M1dag)^-1 M2 phi
# chi = (M1 M1dag)^-1 M2 phi, psi = M1^-1 M2 phi
# dS = (phi, dM2dag chi) + (chi, dM2 phi) - (chi, dM1 M1dag chi) - (chi, M1 dM1dag chi)
class two_flavor_ratio_base(action_base):
    def __init__(self, M, inverter, operator):
        assert len(M) == 2
        super().__init__(M, inverter, operator)

    def __call__(self, fields):
        M1, M2, U, phi = self._updated(fields)

        psi = g.lattice(phi)
        psi @= self.operator.M(M2) * phi
        chi = g.lattice(phi)
        chi @= self.inverter(self.operator.MMdag(M1)) * psi
        return g.inner_product(psi, chi).real

    def draw(self, fields, rng):
        M1, M2, U, phi = self._updated(fields)

        eta = g.lattice(phi)
        rng.cnormal(eta, sigma=2.0**-0.5)  # 1/sqrt(2)

        # phi^dag M2dag (M1 M1dag)^-1 M2 phi
        # eta = M1^-1 M2 phi
        chi = g.lattice(phi)
        chi @= self.inverter(self.operator.MMdag(M2)) * self.operator.M(M1) * eta
        phi @= self.operator.Mdag(M2) * chi
        return g.norm2(eta)

    def gradient(self, fields, dfields):
        M1, M2, U, phi = self._updated(fields)

        frc = self._allocate_force(U)

        psi = g.lattice(phi)
        psi @= self.operator.M(M2) * phi
        chi = g.lattice(phi)
        chi @= self.inverter(self.operator.MMdag(M1)) * psi
        psi @= self.operator.Mdag(M1) * chi

        self._accumulate(frc, self.operator.Mderiv(M2)(chi, phi), +1)
        self._accumulate(frc, self.operator.MderivDag(M2)(phi, chi), +1)
        self._accumulate(frc, self.operator.Mderiv(M1)(chi, psi), -1)
        self._accumulate(frc, self.operator.MderivDag(M1)(psi, chi), -1)

        dS = []
        for f in dfields:
            mu = fields.index(f)
            if mu < len(fields) - 1:
                dS.append(g.qcd.gauge.project.traceless_hermitian(frc[mu]))
            else:
                raise Exception("not implemented")
        return dS


class two_flavor_ratio(two_flavor_ratio_base):
    def __init__(self, M, inverter):
        super().__init__(M, inverter, MMdag)


class two_flavor_ratio_evenodd_schur(two_flavor_ratio_base):
    def __init__(self, M, inverter):
        super().__init__(M, inverter, MMdag_evenodd())
