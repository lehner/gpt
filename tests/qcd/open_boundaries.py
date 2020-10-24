#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Test open boundary conditions
#
import gpt as g
import numpy as np
import sys

# setup rng, mute
g.default.set_verbose("random", False)
rng = g.random("open_boundaries")

# load configuration
U = g.qcd.gauge.random(g.grid([8, 8, 8, 16], g.double), rng)
# U = g.load("/home/rid04246/d2/configs/openqcd/test_16x8n110")
g.message("Plaquette:", g.qcd.gauge.plaquette(U))

# quark
# w = g.qcd.fermion.wilson_clover(
w = g.qcd.fermion.reference.wilson(
    U,
    {
        "kappa": 0.11,
        "csw_r": 1.5,
        "csw_t": 1.5,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, 0.0],
        "cF": 1.3,
    },
)

# default grid
grid = U[0].grid

# create point source, destination
src, dst = g.mspincolor(grid), g.mspincolor(grid)
g.create.point(src, [0, 0, 0, 2])

# abbreviations
i = g.algorithms.inverter
p = g.qcd.fermion.preconditioner

# build solver
cg = i.cg({"eps": 1e-12, "maxiter": 1000})
fgmres = i.fgmres({"eps": 1e-12, "maxiter": 1000})
# slv = i.preconditioned(p.eo2_ne(parity=g.odd), cg) # not cg because Mdag not in reference
slv = fgmres

# calculate propagator
dst @= w.propagator(slv) * src

# two point
corr_2pt = g.slice(g.trace(dst * g.adj(dst)), 3)

# reference
corr_2pt_ref = [
    0.0,
    0.02400406370732104,
    0.6937893453940418,
    0.029038375810865238,
    0.0031216024320516755,
    0.0004785714579322768,
    0.00010094651736288815,
    2.647433330942393e-05,
    7.71992079324806e-06,
    2.3878280342528085e-06,
    7.66326124044683e-07,
    2.5018245407518893e-07,
    8.277640242454872e-08,
    2.7352263656657104e-08,
    6.4311303478774405e-09,
    0.0,
]
# corr_2pt_ref = [  # the one for test_16x8n110 from chroma
#     0.0,
#     0.0232014984431081,
#     0.760212289593613,
#     0.0250445220762833,
#     0.00164287923647052,
#     0.000113657795962742,
#     7.85943713182201e-06,
#     4.92789424316055e-07,
#     3.08721069096737e-08,
#     1.95085023880033e-09,
#     1.15143939616122e-10,
#     7.78273259347072e-12,
#     4.88365539089639e-13,
#     3.25681805328975e-14,
#     1.72080431810457e-15,
#     0.0,
# ]

# output
eps = 0.0
for t in range(len(corr_2pt_ref)):
    eps += (corr_2pt[t].real - corr_2pt_ref[t]) ** 2.0
    g.message(f"corr: {t} {corr_2pt[t].real} {corr_2pt_ref[t]}")
eps = eps ** 0.5 / len(corr_2pt_ref)
g.message(f"eps: {eps}")
assert eps <= 1e-15
g.message("Test successful")
