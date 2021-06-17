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

# 
#   ^mu
#   |___> nu
# 
#  Plaquette action force
#  naive implementation: 2 cshifts and 2 matrix mul for each unoriented mu,nu (=12) for pos/neg
#       total of 48 cshifts and 48 matrix mul
#
#  openQCD, for each oriented mu,nu (=6 planes)
#  w[0] = U[nu](x+mu) * adj(U[mu](x+nu))
#  w[1] = adj(U[nu](x)) * U[mu](x)
#  stpos[nu](x+mu) = w[0] * w[1]
#  stneg[mu](x+nu) = w[1] * w[0]
#  
#  w[1] = w[0] * adj(U[nu](x))
#  stpos[mu](x) = U[mu](x) * w[1]
#  stneg[nu](x) = stpos[mu](x)
#


class symanzik:
    def __init__(self, U, beta, c0):
        self.U = U
        self.grid = U[0].grid
        self.Nd = len(U)
        self.Nc = U[0].otype.Nc
        self.beta = beta
        self.g0inv = beta * 0.5 / self.Nc
        self.c0 = c0

        assert(self.c0 > 0.0)
        self.c1 = (1.0 - c0) / 8.0

        self.staples = []
        self.act_staples = []

        for mu in range(self.Nd):
            act_paths = []
            paths = []
            for nu in range(self.Nd):
                if nu==mu:
                    continue
                    
                #  __
                # .__|  this requires to be multiplied by U[mu](x)^dag
                #       the contribution to the action is U[mu](x) * adj(staple)
                paths.append(
                    gpt.qcd.gauge.path().f(nu,1).f(mu,1).b(nu,1)) 
                if (nu>mu):
                    act_paths.append(paths[-1])
                #  __
                # |__. this requires U[mu](x)^dag as right multiplier
                #      the contrib to the action is staple * adj(U[mu](x))
                paths.append(
                    gpt.qcd.gauge.path().b(nu,1).f(mu,1).f(nu,1))
                
                #if self.c0 != 1.0:
            self.staples.append( gpt.qcd.gauge.transport(self.U, paths) )
            self.act_staples.append( gpt.qcd.gauge.transport(self.U, act_paths) )
        
        self.frc = []
        for umu in U:
            self.frc.append(gpt.lattice(umu))
        
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
                * gpt.adj(self.U[nu] * gpt.cshift(self.U[mu], nu, 1))
            )
        else:
            # tmp = gpt.cshift(self.U[nu], nu, -1)
            # return gpt.adj(gpt.cshift(tmp,mu,1)) * gpt.adj(gpt.cshift(self.U[mu], nu, -1)) * tmp
            tmp = gpt.cshift(self.U[nu], mu, 1)
            return gpt.cshift(gpt.adj(self.U[mu] * tmp) * self.U[nu], nu, -1)

    def __call__(self):
        #act = 0.0
        trU = gpt.complex(self.grid)
        trU[:] = 0
        #Nc = gpt.complex(self.grid)
        #Nc[:] = self.Nc
        #staple = gpt.lattice(self.U[0])
        
        for mu in range(self.Nd):
#             st = self.act_staples[mu](self.U)
            
            for nu in range(mu + 1, self.Nd):
#                 trU += self.c0 * gpt.trace(self.U[mu] * gpt.adj(next(st)))
        
        
#         oriented = self.Nd * (self.Nd-1) / 2.0
#         act = 2.0 * self.grid.fsites * self.Nc * oriented * (self.c0 + self.c1*2.0)
#         act -= 2.0 * gpt.sum(trU).real

                # __
                # __|
                st = self.__staple(mu, nu, True)
                trU @= self.c0 * (Nc - gpt.trace(self.U[mu] * st))

                if self.c0 != 1.0:
                   # __ __
                   # __ __|
                   v = (
                       gpt.cshift(self.U[nu], mu, 1)
                       * gpt.cshift(st, nu, 1)
                       * gpt.adj(self.U[nu])
                   )
                   trU += self.c1 * (Nc - gpt.trace(self.U[mu] * v))

                   #  _
                   # | |    mu
                   #  _|    |__>nu
                   st = gpt.adj(self.__staple(nu, mu, True))
                   v = gpt.cshift(st, mu, 1) * gpt.adj(
                       self.U[nu] * gpt.cshift(self.U[mu], nu, 1)
                   )
                   trU += self.c1 * (Nc - gpt.trace(self.U[mu] * v))

                act += 2.0 * gpt.sum(trU[0]).real
        act *= self.g0inv
        return act

    def hot_start(self, rng):
        # U = exp(Ta * ca), with ca from normal distrib.
        for u in self.U:
            rng.element(u, normal=True)
            

    def setup_force(self):
        for _f in self.frc:
            _f[:] = 0

        w = [gpt.lattice(self.U[0]) for i in range(2)]
        x = gpt.lattice(self.U[0])
        
        for mu in range(self.Nd):
            for nu in range(mu+1,self.Nd):
                w[0] @= gpt.cshift(self.U[nu],mu,1) * gpt.adj(gpt.cshift(self.U[mu],nu,1))
                w[1] @= gpt.adj(self.U[nu]) * self.U[mu]

                x @= gpt.cshift(w[0] * w[1],mu,-1)
#                 self.sun2alg(x)
                self.frc[nu] += x
                
                x @= gpt.cshift(w[1] * w[0],nu,-1)
#                 self.sun2alg(x)
                self.frc[mu] -= x
                                
                x @= self.U[mu] * w[0] * gpt.adj(self.U[nu])
#                 self.sun2alg(x)
                self.frc[mu] += x
                self.frc[nu] -= x
        
        for mu in range(self.Nd):
            self.frc[mu] *= self.c0

        if (self.c0 != 0.0):
            self.setup_force_rectangles()
            
        for mu in range(self.Nd):
            self.sun2alg(self.frc[mu])
            
    def setup_force_rectangles(self):
        for mu in range(self.Nd):
            for nu in range(mu+1,self.Nd):
                    # __ __
                    # __ __|
                    tmp2 = gpt.cshift(self.U[nu], mu, 1)
                    frc[mu] += (
                        self.c1
                        * self.U[mu]
                        * tmp2
                        * gpt.cshift(tmp, nu, 1)
                        * gpt.adj(self.U[nu])
                    )

                    #  _
                    # | |    mu
                    #  _|    |__>nu
                    tmp = gpt.cshift(self.__staple(nu, mu, True), mu, 1)
                    frc[mu] += (
                        self.c1
                        * self.U[mu]
                        * gpt.adj(tmp)
                        * gpt.adj(self.U[nu] * gpt.cshift(self.U[mu], nu, 1))
                    )

                    #  _
                    # | |    mu
                    # |_     |__>nu
                    frc[mu] -= self.c1 * gpt.cshift(
                        tmp * gpt.adj(self.U[mu]) * self.U[nu], nu, -1
                    ) * gpt.adj(self.U[mu])

                    #  _
                    #   |    mu
                    # |_|    |__>nu
                    tmp = self.__staple(nu, mu, False)
                    frc[mu] += (
                        self.c1
                        * self.U[mu]
                        * tmp2
                        * gpt.adj(gpt.cshift(self.U[mu], nu, 1))
                        * tmp
                    )

                    #  _
                    # |      mu
                    # |_|    |__>nu
                    frc[mu] -= self.c1 * gpt.adj(
                        self.U[mu] * gpt.cshift(
                            tmp * self.U[mu] * tmp2, nu, -1
                        )
                    )

                    #  __ __
                    # |__ __
                    frc[mu] -= self.c1 * gpt.cshift(
                        gpt.adj(tmp2) * tmp * self.U[nu], nu, -1,
                    ) * gpt.adj(self.U[mu])

        return [stpos, stneg]

    def compute_staples_0(self, mu):
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
                    tmp2 = gpt.cshift(self.U[nu], mu, 1)
                    stpos += (
                        self.c1
                        * tmp2
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
                        * tmp2
                        * gpt.adj(gpt.cshift(self.U[mu], nu, 1))
                        * tmp
                    )

                    #  _
                    # |      mu
                    # |_|    |__>nu
                    stneg += self.c1 * gpt.adj(
                        gpt.cshift(
                            tmp * self.U[mu] * tmp2, nu, -1
                        )
                    )

                    #  __
                    # |__
                    tmp = self.__staple(mu, nu, False)
                    stneg += self.c0 * tmp

                    #  __ __
                    # |__ __
                    stneg += self.c1 * gpt.cshift(
                        gpt.adj(tmp2) * tmp * self.U[nu], nu, -1,
                    )

        return [stpos, stneg]

#     def compute_staples_1(self, mu):
#         stpos = gpt.lattice(self.U[0])
#         stneg = gpt.lattice(self.U[0])
#         stpos[:] = 0
#         stneg[:] = 0

#         st = self.staples[mu](self.U)
#         for nu in range(self.Nd):
#             if nu==mu:
#                 continue
#             stpos += gpt.adj(next(st));
#             stneg += next(st);
    
#         return [stpos, stneg]
    
        
    def sun2alg(self, link):
        # U -> 0.5*(U - U^dag) - 0.5/N * tr(U-U^dag)
        link -= gpt.adj(link)
        tr = gpt.eval(gpt.trace(link)) / self.Nc
        link -= gpt.mcolor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * tr
        link *= 0.5

    def force(self, field):
        #frc = gpt.lattice(self.U[0])
        mu = -1
                        
        for U in self.U:
            if U.v_obj == field.v_obj:
                mu = self.U.index(U)
                #[stpos, stneg] = self.compute_staples(mu)
                #frc @= U * stpos
                #frc -= gpt.adj(U * stneg)
                #[stpos, stneg] = self.compute_staples(mu)
                #frc @= (U * stpos) - (stneg * gpt.adj(U)) 
                #frc @= stpos[mu] - stneg[mu]
                
        if mu==-1:
            raise Exception
            
        return self.frc[mu] * self.g0inv


class wilson(symanzik):
    def __init__(self, U, beta):
        symanzik.__init__(self, U, beta, 1.0)


class luscher_weisz(symanzik):
    def __init__(self, U, beta):
        symanzik.__init__(self, U, beta, 5.0 / 3.0)


class iwasaki(symanzik):
    def __init__(self, U, beta):
        symanzik.__init__(self, U, beta, 3.648)
