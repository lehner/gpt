#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g
import numpy as np

# load configuration
# U = g.load("/hpcgpfs01/work/clehner/configs/openQCD/A250t000n54")
U = g.qcd.gauge.random(g.grid([24, 24, 24, 32], g.double), g.random("T"))

# do everything in single-precision
U = g.convert(U, g.single)

# use the gauge configuration grid
grid = U[0].grid
L = np.array(grid.fdimensions)

# quark
w = g.qcd.fermion.wilson_clover(
    U,
    {
        "kappa": 0.137,
        "csw_r": 0,
        "csw_t": 0,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, -1.0],
    },
)

# create point source
src = g.mspincolor(grid)
g.create.point(src, [0, 0, 0, 0])

# even-odd preconditioned matrix
eo = g.qcd.fermion.preconditioner.eo2_ne()

# build solver
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-6, "maxiter": 1000})
propagator = w.propagator(inv.preconditioned(pc.eo1_ne(), cg))

# propagator
dst = g.mspincolor(grid)
dst @= propagator * src

# momentum
p = 2.0 * np.pi * np.array([1, 0, 0, 0]) / L
P = g.exp_ixp(p)

# operators
G_src = g.gamma[5] * P
G_snk = g.gamma[5] * g.adj(P)
G_op = g.gamma["T"]

# 2pt
correlator_2pt = g.slice(g.trace(G_src * g.gamma[5] * g.adj(dst) * g.gamma[5] * G_snk * dst), 3)

# sequential solve through t=8
t_op = 8
src_seq = g.lattice(src)
src_seq[:] = 0
src_seq[:, :, :, t_op] = dst[:, :, :, t_op]

# create seq prop with gamma_T operator
dst_seq = g.lattice(src_seq)
src_seq @= G_op * src_seq
dst_seq @= propagator * src_seq

# 3pt
correlator_3pt = g.slice(g.trace(G_src * g.gamma[5] * g.adj(dst) * g.gamma[5] * G_snk * dst_seq), 3)

# output
for t in range(len(correlator_2pt)):
    g.message(t, correlator_2pt[t].real, (correlator_3pt[t] / correlator_2pt[t]).real)
