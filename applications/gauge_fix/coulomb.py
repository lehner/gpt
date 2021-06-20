#!/usr/bin/env python3
#
# Author: Christoph Lehner 2021
#
import gpt as g

U = g.load("/p/project/gm2dwf/configs/64I/ckpoint_lat.Coulomb.1200")

# split in time
Usep = [g.separate(u,3) for u in U[0:3]]
Nt = U[0].grid.gdimensions[3]

# gradient descent
gd = g.algorithms.optimize.gradient_descent(maxiter=100000, eps=1e-7, step=0.1)

# Coulomb functional on each time-slice
for t in range(Nt):
    Ut = [Usep[mu][t] for mu in range(3)]
    V = g.identity(Ut[0])
    g.random("f").element(V)
    f, df = g.qcd.gauge.fix.landau(Ut)
    fa_df = g.algorithms.optimize.fourier_accelerate.inverse_phat_square(V.grid, df)

    gd(f, fa_df)(V)
    
    dfv = df(V)
    theta = g.norm2(dfv) / V.grid.gsites / dfv.otype.Nc
    g.message(t, theta)
    g.message(V[0,0,0])
