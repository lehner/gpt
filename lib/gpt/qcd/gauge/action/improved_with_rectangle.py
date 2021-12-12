#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2021  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


class improved_with_rectangle(base):
    def __init__(self, beta, c1, c0=None):
        self.beta = beta
        self.c1 = c1
        if c0 is None:
            c0 = 1.0 - 8 * c1
        self.c0 = c0
        self.staple_1x1 = {}
        self.staple_2x1 = {}
        self.__name__ = f"improved_with_rectangle({beta},{c1})"

    def __call__(self, U):
        Nd = len(U)
        vol = U[0].grid.gsites
        P = g.qcd.gauge.plaquette(U)
        R = g.qcd.gauge.rectangle(
            U, [[(mu, 2, nu, 1) for mu in range(Nd) for nu in range(Nd) if mu != nu]]
        )
        return (
            vol
            * self.beta
            * (
                self.c0 * (1.0 - P) * (Nd - 1) * Nd / 2.0
                + self.c1 * (1.0 - R) * (Nd - 1) * Nd
            )
        )

    def staple(self, U, mu):
        Nd = len(U)
        st = g.lattice(U[0])
        st[:] = 0

        O = [nu for nu in range(Nd) if nu != mu]
        path = g.qcd.gauge.path
        if (Nd, mu) not in self.staple_1x1:
            p = []
            p += [path().f(nu).f(mu).b(nu) for nu in O]
            p += [path().b(nu).f(mu).f(nu) for nu in O]
            self.staple_1x1[(Nd, mu)] = g.qcd.gauge.transport(U, p)

        if (Nd, mu) not in self.staple_2x1:
            p = []
            p += [path().f(nu, 2).f(mu).b(nu, 2) for nu in O]
            p += [path().b(nu, 2).f(mu).f(nu, 2) for nu in O]
            p += [path().f(nu).f(mu, 2).b(nu).b(mu) for nu in O]
            p += [path().b(nu).f(mu, 2).f(nu).b(mu) for nu in O]
            p += [path().b(mu).b(nu).f(mu, 2).f(nu) for nu in O]
            p += [path().b(mu).f(nu).f(mu, 2).b(nu) for nu in O]
            self.staple_2x1[(Nd, mu)] = g.qcd.gauge.transport(U, p)

        for s in self.staple_1x1[(Nd, mu)](U):
            st += (self.beta * self.c0 / U[0].otype.shape[0]) * s

        for s in self.staple_2x1[(Nd, mu)](U):
            st += (self.beta * self.c1 / U[0].otype.shape[0]) * s

        return st


class iwasaki(improved_with_rectangle):
    def __init__(self, beta):
        super().__init__(beta, -0.331)


class dbw2(improved_with_rectangle):
    def __init__(self, beta):
        super().__init__(beta, -1.40686)


class symanzik(improved_with_rectangle):
    def __init__(self, beta):
        super().__init__(beta, -1.0 / 12.0)
