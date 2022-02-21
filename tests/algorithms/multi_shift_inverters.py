#!/usr/bin/env python3
#
# Authors: Mattia Bruno 2020
#          Christoph Lehner 2020
#
import gpt as g
import numpy as np

# load configuration
precision = g.double
U = g.qcd.gauge.random(g.grid([8, 8, 8, 16], precision), g.random("test"))

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
src[:] = 0
src[0, 1, 0, 0] = g.vspincolor([[1] * 3] * 4)

shifts = [0.1, 0.05, 0.01]

# inverters
inv = g.algorithms.inverter
cg = inv.cg({"eps": 1e-6, "maxiter": 1024})
g.default.set_verbose("cg")
g.default.set_verbose("multi_shift_cg")
g.default.set_verbose("multi_shift_inverter")

def MMdag(dst, src):
    dst @= w * g.adj(w) * src
    
minv = g.algorithms.multi_inverter
mscg = minv.multi_shift_cg({"eps": 1e-6, "maxiter": 1024})
mat_inv = mscg(MMdag, shifts)
dst = mat_inv(src)

g.message(f"MSCG: testing solutions [1 - (MMdag+s)(MMdag+s)^-1] for various shifts")
eps = g.norm2(mat_inv.inv_mat(dst)) - g.norm2(src)
for i, s in enumerate(shifts):
    g.message(f'shift {i} [= {s}] = {eps[i]:e}')

mscg_seq = minv.multi_shift_inverter(cg)
mat_inv = mscg_seq(MMdag, shifts)
dst_seq = mat_inv(src)

g.message(f"MSCG SEQUENTIAL: testing solutions [1 - (MMdag+s)(MMdag+s)^-1] for various shifts")
eps = g.norm2(mat_inv.inv_mat(dst_seq)) - g.norm2(src)
for i, s in enumerate(shifts):
    g.message(f'shift {i} [= {s}] = {eps[i]:e}')

g.message(f"MSCG - MSCG SEQUENTIAL")
for i, s in enumerate(shifts):
    g.message(f'shift {i} [= {s}] = {g.norm2(dst[i] - dst_seq[i]):e}')
