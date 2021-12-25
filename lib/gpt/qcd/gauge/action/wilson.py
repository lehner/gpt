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
from gpt.qcd.gauge.action import base


class wilson(base):
    def __init__(self, beta):
        self.beta = beta
        self.__name__ = f"wilson({beta})"

    def __call__(self, U):
        # Let beta = 2 ndim_repr / g^2
        #
        # S(U) = -beta sum_{mu>nu} Re[Tr[P_{mu,nu}]]/ndim_repr        (only U-dependent part)
        #      = -2/g^2 sum_{mu>nu} Re[Tr[P_{mu,nu}]]
        #      = -1/g^2 sum_{mu,nu} Re[Tr[P_{mu,nu}]]
        #      = -2/g^2 sum_{mu,nu} Re[Tr[staple_{mu,nu}^dag U_mu]]
        #
        # since   P_{mu,nu} = staple_{mu,nu}^dag U_mu + staple_{mu,nu} U_mu^dag = 2 Re[staple^dag * U]
        Nd = len(U)
        vol = U[0].grid.gsites
        return self.beta * (1.0 - g.qcd.gauge.plaquette(U)) * (Nd - 1) * Nd * vol / 2.0

    def staple(self, U, mu):
        st = g.lattice(U[0])
        st[:] = 0
        Nd = len(U)
        for nu in range(Nd):
            if mu != nu:
                st += g.qcd.gauge.staple(U, mu, nu)
        scale = self.beta / U[0].otype.shape[0]
        return g(scale * st)
