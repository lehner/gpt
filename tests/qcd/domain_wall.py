#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys
import time

# load configuration
# U = g.load("/hpcgpfs01/work/clehner/configs/16I_0p01_0p04/ckpoint_lat.IEEE64BIG.1100")
rng = g.random("test")
U = g.qcd.gauge.random(g.grid([8, 8, 8, 8], g.double), rng, scale=0.5)
g.message("Plaquette:", g.qcd.gauge.plaquette(U))

# do everything in single-precision
U = g.convert(U, g.single)

# use the gauge configuration grid
grid = U[0].grid

# mobius <> zmobius domain wall quark
mobius_params = {
    "mass": 0.08,
    "M5": 1.8,
    "b": 1.5,
    "c": 0.5,
    "Ls": 12,
    "boundary_phases": [1.0, 1.0, 1.0, 1.0],
}

qm = g.qcd.fermion.mobius(g.qcd.gauge.unit(grid), mobius_params)

w = g.qcd.fermion.wilson_clover(
    U,
    {
        "mass": qm.params["M5"],
        "csw_r": 0,
        "csw_t": 0,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    },
)

# test operator update
start = g.vspincolor(qm.F_grid)
rng.cnormal(start)
qm_new = g.qcd.fermion.mobius(U, mobius_params)
qm.update(U)
eps2 = g.norm2(qm * start - qm_new * start) / g.norm2(start)
g.message(f"Operator update test: {eps2}")
assert eps2 < 1e-10


# solver
pc = g.qcd.fermion.preconditioner
inv = g.algorithms.inverter
cg = inv.cg({"eps": 1e-4, "maxiter": 1000})


def H5_denom(dst, src):
    dst @= g.gamma[5] * (2 * src + (qm.params["b"] - qm.params["c"]) * w * src)


inv_H5_denom = cg(H5_denom)


def H5(dst, src):
    # g5 * Dkernel
    # Dkernel = (b+c)*w / (2 + (b-c)*w)
    dst @= inv_H5_denom * w * src
    dst *= qm.params["b"] + qm.params["c"]


# arnoldi to get an idea of entire spectral range of w
start = g.vspincolor(w.F_grid)
start[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
g.default.push_verbose("arnoldi", False)
a = g.algorithms.eigen.arnoldi(Nmin=20, Nmax=20, Nstep=0, Nstop=20, resid=1)
_, evals_H5 = a(H5, start)
g.default.pop_verbose()
g.message(evals_H5)
# H5 spectrum for 16c RBC lattice: [-1.6344227, ..., 1.6344227]
# H5 spectrum for random lattice:  [-1.6442832, ..., 1.6442832]

qz = g.qcd.fermion.zmobius(
    U,
    {
        "mass": 0.08,
        "M5": 1.8,
        "b": 1.0,
        "c": 0.0,
        "omega": [
            0.17661651536320583 + 1j * (0.14907774771612217),
            0.23027432016909377 + 1j * (-0.03530801572584271),
            0.3368765581549033 + 1j * (0),
            0.7305711010541054 + 1j * (0),
            1.1686138337986505 + 1j * (0.3506492418109086),
            1.1686138337986505 + 1j * (-0.3506492418109086),
            0.994175013717952 + 1j * (0),
            0.5029903152251229 + 1j * (0),
            0.23027432016909377 + 1j * (0.03530801572584271),
            0.17661651536320583 + 1j * (-0.14907774771612217),
        ],
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    },
)

# create point source
src = g.mspincolor(grid)
g.create.point(src, [0, 1, 0, 0])

# solver
pc = g.qcd.fermion.preconditioner
inv = g.algorithms.inverter
cg = inv.cg({"eps": 1e-5, "maxiter": 1000})

slv_5d = inv.preconditioned(pc.eo2_ne(), cg)
slv_qm = qm.propagator(slv_5d)
slv_qz = qz.propagator(slv_5d)
slv_madwf = qm.propagator(pc.mixed_dwf(slv_5d, slv_5d, qz))

# propagator
dst_qm = g.mspincolor(grid)
dst_qz = g.mspincolor(grid)

dst_qm @= slv_qm * src
dst_qz @= slv_qz * src

# test madwf
src_sc = rng.cnormal(g.vspincolor(grid))
dst_madwf_sc = g(slv_madwf * src_sc)
dst_dwf_sc = g(slv_qm * src_sc)
eps2 = g.norm2(dst_madwf_sc - dst_dwf_sc) / g.norm2(dst_dwf_sc)
g.message(f"MADWF test: {eps2}")
assert eps2 < 5e-4

# two-point
correlator_qm = g.slice(g.trace(dst_qm * g.adj(dst_qm)), 3)
correlator_qz = g.slice(g.trace(dst_qz * g.adj(dst_qz)), 3)
correlator_ref = [
    0.4876481890678406,
    0.14691542088985443,
    0.020957840606570244,
    0.007922700606286526,
    0.005805496126413345,
    0.00794452615082264,
    0.021211711689829826,
    0.14737030863761902,
]

# output
eps_qm = 0.0
eps_qz = 0.0
for t in range(len(correlator_ref)):
    eps_qm += (correlator_qm[t].real - correlator_ref[t]) ** 2.0
    eps_qz += (correlator_qz[t].real - correlator_ref[t]) ** 2.0
    g.message(t, correlator_qm[t].real, correlator_qz[t].real, correlator_ref[t])
eps_qm = eps_qm ** 0.5 / len(correlator_ref)
eps_qz = eps_qz ** 0.5 / len(correlator_ref)
g.message("Test results: %g %g" % (eps_qm, eps_qz))
assert eps_qm < 1e-5
assert eps_qz < 5e-4
g.message("Test successful")
