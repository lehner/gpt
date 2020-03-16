#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys

# will need cg later
cg=g.algorithms.iterative.cg

# load configuration
U = g.load("/hpcgpfs01/work/clehner/configs/16I_0p01_0p04/ckpoint_lat.IEEE64BIG.1100")

# report plaquette
g.message("Plaquette:", g.qcd.gauge.plaquette(U))

# do everything in single-precision
U = g.convert(U, g.single)

# use the gauge configuration grid
grid=U[0].grid

# wilson operator
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

# create point source
src=g.mspincolor(grid)
src[:]=0
src[0,0,0,0]=g.mspincolor(np.multiply.outer( np.identity(4) , np.identity(3) ))

# create wilson operator
w=wilson(0.137,U)

# propagator
# sum_n D^-1 vn vn^dag src = D^-1 vn (src^dag vn)^dag
dst_sc,src_sc=g.vspincolor(grid),g.vspincolor(grid)
dst=g.mspincolor(grid)
dst[:]=0

for s in range(4):
    for c in range(3):

        g.qcd.prop_to_ferm(src_sc,src,s,c)

        dst_sc @= g.gamma[5] * src_sc
        w.G5M(src_sc,dst_sc)        

        dst_sc[:]=0

        cg(lambda i,o: w.G5Msqr(o,i),src_sc,dst_sc,1e-6,1000)
        
        g.qcd.ferm_to_prop(dst,dst_sc,s,c)

# two-point
correlator=g.slice(g.trace(dst*g.adj(dst)),3)

# output
for t,c in enumerate(correlator):
    print(t,c.real)

