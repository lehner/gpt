#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt as g

class wilson:
    # M = sum_mu gamma[mu]*D[mu] + m0 - 1/2 sum_mu D^2[mu]
    # m0 + 4 = 1/2/kappa
    def __init__(self, kappa, U):
        self.kappa = kappa
        self.U = U
        self.Udag = [ g.eval(g.adj(u)) for u in U ]

    def Meooe(self, dst, src):
        assert(dst != src)
        dst[:]=0
        for mu in range(4):
            src_plus = self.U[mu]*g.cshift(src,mu,+1)
            dst += 1./2.*g.gamma[mu]*src_plus - 1./2.*src_plus

            src_minus = g.cshift(self.Udag[mu]*src,mu,-1)
            dst += -1./2.*g.gamma[mu]*src_minus - 1./2.*src_minus

    def Mooee(self, dst, src):
        assert(dst != src)
        dst @= 1./2.*1./self.kappa * src

    def M(self, dst, src):
        assert(dst != src)
        t=g.lattice(dst)
        self.Meooe(t,src)
        self.Mooee(dst,src)
        dst += t

    def G5M(self, dst, src):
        assert(dst != src)
        self.M(dst,src)
        dst @= g.gamma[5] * dst

    def G5Msqr(self, dst, src):
        assert(dst != src)
        t=g.lattice(dst)
        self.G5M(t,src)
        self.G5M(dst,t)
