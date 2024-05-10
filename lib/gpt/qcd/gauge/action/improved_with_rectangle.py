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
        self.cache = {}
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
            * (self.c0 * (1.0 - P) * (Nd - 1) * Nd / 2.0 + self.c1 * (1.0 - R) * (Nd - 1) * Nd)
        )

    def staples(self, U, mu_target=None):
        Nd = len(U)

        if mu_target not in self.cache:
            code = []
            Nret = 0
            path = g.path
            for mu in range(Nd):
                if mu_target is not None and mu_target != mu:
                    continue

                O = [nu for nu in range(Nd) if nu != mu]
                code_mu = []
                for nu in O:
                    # 1x1 staples
                    s0 = self.beta * self.c0 / U[0].otype.shape[0]
                    code_mu.append(
                        (Nret, -1 if len(code_mu) == 0 else Nret, s0, path().f(nu).f(mu).b(nu))
                    )
                    code_mu.append(
                        (Nret, -1 if len(code_mu) == 0 else Nret, s0, path().b(nu).f(mu).f(nu))
                    )

                    # 2x1 staples
                    s1 = self.beta * self.c1 / U[0].otype.shape[0]
                    dst = -1 if len(code_mu) == 0 else Nret
                    code_mu.append(
                        (
                            Nret,
                            dst,
                            s1,
                            path().f(nu, 2).f(mu).b(nu, 2),
                        )
                    )
                    dst = Nret
                    code_mu.append(
                        (
                            Nret,
                            Nret,
                            s1,
                            path().b(nu, 2).f(mu).f(nu, 2),
                        )
                    )
                    code_mu.append(
                        (
                            Nret,
                            Nret,
                            s1,
                            path().f(nu).f(mu, 2).b(nu).b(mu),
                        )
                    )
                    code_mu.append((Nret, Nret, s1, path().b(nu).f(mu, 2).f(nu).b(mu)))
                    code_mu.append((Nret, Nret, s1, path().b(mu).b(nu).f(mu, 2).f(nu)))
                    code_mu.append((Nret, Nret, s1, path().b(mu).f(nu).f(mu, 2).b(nu)))

                code = code + code_mu
                Nret += 1

            self.cache[mu_target] = g.parallel_transport_matrix(U, code, Nret)

        ret = self.cache[mu_target](U)
        if mu_target is not None:
            ret = [ret]
        return ret


class iwasaki(improved_with_rectangle):
    def __init__(self, beta):
        super().__init__(beta, -0.331)


class dbw2(improved_with_rectangle):
    def __init__(self, beta):
        super().__init__(beta, -1.40686)


class symanzik(improved_with_rectangle):
    def __init__(self, beta):
        super().__init__(beta, -1.0 / 12.0)
