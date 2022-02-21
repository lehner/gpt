#!/usr/bin/env python3
#
# Authors: Mattia Bruno 2022
#          Christoph Lehner 2022
#
import numpy
import gpt as g
import numpy as np

# load configuration
precision = g.double
rng = g.random("rational")
U = g.qcd.gauge.random(g.grid([8, 8, 8, 16], precision), rng)

# use the gauge configuration grid
grid = U[0].grid

# quark
w = g.qcd.fermion.wilson_clover(
    U,
    {
        "kappa": 0.13565,
        "csw_r": 2.0171 / 2.0,  # for now test with very heavy quark
        "csw_t": 2.0171 / 2.0,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    },
)

# create point source
src = g.vspincolor(grid)
rng.cnormal(src)

# inverters
inv = g.algorithms.inverter
g.default.set_verbose("multi_shift_cg")

def MMdag(dst, src):
    dst @= w * g.adj(w) * src
    
minv = g.algorithms.multi_inverter
mscg = minv.multi_shift_cg({"eps": 1e-6, "maxiter": 1024})

rat = g.algorithms.rational

np = 12
zol = rat.zolotarev_inverse_square_root(0.1, 4.5, np)
g.message(zol)
for x in numpy.arange(0.1, 4.5, 0.05):
    assert zol.relative_error(x) < 1e-9


r = rat.rational_polynomial(zol.num, zol.poles, mscg)
g.message(r)
rr = r(MMdag)

psi, phi = g.lattice(src), g.lattice(src)
psi @= rr * src

inv = g.algorithms.inverter
cg = inv.cg({"eps": 1e-12, "maxiter": 1024})
g.default.set_verbose("cg")

def MMdag_shift(dst, src, s):
    dst @= w * g.adj(w) * src
    dst += s * src

phi @= src
for i in range(np):
    g.message(f" (MMdag + {zol.num[i]})/(MMdag + {zol.poles[i]})")
    mat_inv = cg(lambda dst, src: MMdag_shift(dst, src, zol.poles[i]))
    tmp = mat_inv(phi)
    MMdag_shift(phi, tmp, zol.num[i])
    
g.message("Testing rat_polynomial against exact calculation")
g.message(f"  = {g.inner_product(src, phi).real - g.inner_product(src, psi).real:e}")
