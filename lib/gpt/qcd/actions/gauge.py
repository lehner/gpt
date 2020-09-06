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


class symanzik:
    def __init__(self, U, beta, c0):
        self.U = U
        self.grid = U[0].grid
        self.Nd = len(U)
        self.Nc = U[0].otype.Nc
        self.beta = beta
        self.g0inv = beta * 0.5 / self.Nc
        self.c0 = c0
        assert self.c0 > 0.0
        self.c1 = (1.0 - c0) / 8.0

    def __staple(self, mu, nu, pos):
        #  __     __
        #    |   |     ^mu
        #  __|   |__   |___> nu
        #
        # U[mu](x) -- U[nu](x+mu)*adj(U[mu](x+nu))*adj(U[nu](x))
        # adj(U[nu](x-nu+mu))*adj(U[mu](x-nu))*U[nu](x-nu)
        if pos:
            return (
                gpt.cshift(self.U[nu], mu, 1)
                * gpt.adj(gpt.cshift(self.U[mu], nu, 1))
                * gpt.adj(self.U[nu])
            )
        else:
            # tmp = gpt.cshift(self.U[nu], nu, -1)
            # return gpt.adj(gpt.cshift(tmp,mu,1)) * gpt.adj(gpt.cshift(self.U[mu], nu, -1)) * tmp
            tmp = gpt.cshift(self.U[nu], mu, 1)
            return gpt.cshift(gpt.adj(self.U[mu] * tmp) * self.U[nu], nu, -1)

    def __call__(self):
        act = 0.0
        trU = [gpt.complex(self.grid) for _ in range(3)]
        Nc = gpt.complex(self.grid)
        Nc[:] = self.Nc

        for mu in range(self.Nd):
            for nu in range(mu + 1, self.Nd):
                # __
                # __|
                st = self.__staple(mu, nu, True)
                trU[0] @= self.c0 * (Nc - gpt.trace(self.U[mu] * st))

                if self.c0 != 1.0:
                    # __ __
                    # __ __|
                    v = (
                        gpt.cshift(self.U[nu], mu, 1)
                        * gpt.cshift(st, nu, 1)
                        * gpt.adj(self.U[nu])
                    )
                    trU[1] @= self.c1 * (Nc - gpt.trace(self.U[mu] * v))

                    #  _
                    # | |    mu
                    #  _|    |__>nu
                    st = gpt.adj(self.__staple(nu, mu, True))
                    v = gpt.cshift(st, mu, 1) * gpt.adj(
                        self.U[nu] * gpt.cshift(self.U[mu], nu, 1)
                    )
                    trU[2] @= self.c1 * (Nc - gpt.trace(self.U[mu] * v))

                if self.c0 == 1.0:
                    act += 2.0 * gpt.sum(trU[0]).real
                else:
                    act += 2.0 * gpt.sum(trU[0] + trU[1] + trU[2]).real
        act *= self.g0inv
        return act

    def hot_start(self, rng):
        ca = gpt.complex(self.U[0].grid)
        ta = gpt.lattice(self.U[0])
        lie = gpt.lattice(self.U[0])

        for u in self.U:
            lie[:] = 0
            for g in u.otype.generators(u.grid.precision.complex_dtype):
                rng.normal(ca, {"mu": 0.0, "sigma": 1.0})
                ta[:] = g
                lie += 1j * ca * ta
            u @= gpt.core.matrix.exp(lie)

    def setup_force(self):
        pass

    def compute_staples(self, mu):
        stpos = gpt.lattice(self.U[0])
        stneg = gpt.lattice(self.U[0])
        stpos[:] = 0
        stneg[:] = 0

        for nu in range(self.Nd):
            if nu != mu:
                if self.c0 == 1.0:
                    stpos += self.__staple(mu, nu, True)
                    stneg += self.__staple(mu, nu, False)
                else:
                    # __
                    # __|
                    tmp = self.__staple(mu, nu, True)
                    stpos += self.c0 * tmp

                    # __ __
                    # __ __|
                    stpos += (
                        self.c1
                        * gpt.cshift(self.U[nu], mu, 1)
                        * gpt.cshift(tmp, nu, 1)
                        * gpt.adj(self.U[nu])
                    )

                    #  _
                    # | |    mu
                    #  _|    |__>nu
                    tmp = gpt.cshift(self.__staple(nu, mu, True), mu, 1)
                    stpos += (
                        self.c1
                        * gpt.adj(tmp)
                        * gpt.adj(self.U[nu] * gpt.cshift(self.U[mu], nu, 1))
                    )

                    #  _
                    # | |    mu
                    # |_     |__>nu
                    stneg += self.c1 * gpt.cshift(
                        tmp * gpt.adj(self.U[mu]) * self.U[nu], nu, -1
                    )

                    #  _
                    #   |    mu
                    # |_|    |__>nu
                    tmp = self.__staple(nu, mu, False)
                    stpos += (
                        self.c1
                        * gpt.cshift(self.U[nu], mu, 1)
                        * gpt.adj(gpt.cshift(self.U[mu], nu, 1))
                        * tmp
                    )

                    #  _
                    # |      mu
                    # |_|    |__>nu
                    stneg += self.c1 * gpt.adj(
                        gpt.cshift(
                            tmp * self.U[mu] * gpt.cshift(self.U[nu], mu, 1), nu, -1
                        )
                    )

                    #  __
                    # |__
                    tmp = self.__staple(mu, nu, False)
                    stneg += self.c0 * tmp

                    #  __ __
                    # |__ __
                    stneg += self.c1 * gpt.cshift(
                        gpt.adj(gpt.cshift(self.U[nu], mu, 1)) * tmp * self.U[nu],
                        nu,
                        -1,
                    )

        return [stpos, stneg]

    def sun2alg(self, link):
        # U -> 0.5*(U - U^dag) - 0.5/N * tr(U-U^dag)
        link -= gpt.adj(link)
        tr = gpt.eval(gpt.trace(link)) / self.Nc
        tmp = gpt.lattice(link)
        tmp[:] = gpt.mcolor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        link -= tmp * tr
        link *= 0.5

    def force(self, field):
        frc = gpt.lattice(self.U[0])
        for U in self.U:
            if U.v_obj == field.v_obj:
                mu = self.U.index(U)
                [stpos, stneg] = self.compute_staples(mu)
                frc @= U * stpos
                frc -= gpt.adj(U * stneg)

        self.sun2alg(frc)
        return frc * self.g0inv


class wilson(symanzik):
    def __init__(self, U, beta):
        symanzik.__init__(self, U, beta, 1.0)


class luscher_weisz(symanzik):
    def __init__(self, U, beta):
        symanzik.__init__(self, U, beta, 5.0 / 3.0)


class iwasaki(symanzik):
    def __init__(self, U, beta):
        symanzik.__init__(self, U, beta, 3.648)
