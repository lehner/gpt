#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys, cmath

# load configuration
rng = g.random("test")
U = g.qcd.gauge.random(g.grid([8, 8, 8, 16], g.double), rng)

# do everything in single-precision
U = g.convert(U, g.single)

# plaquette
g.message("Plaquette:", g.qcd.gauge.plaquette(U))

# use the gauge configuration grid
grid = U[0].grid

# wilson parameters
p = {
    "kappa": 0.137,
    "csw_r": 0,
    "csw_t": 0,
    "xi_0": 1,
    "nu": 1,
    "isAnisotropic": False,
    "boundary_phases": [cmath.exp(1j), cmath.exp(2j), cmath.exp(3j), cmath.exp(4j)],
}

# pf=g.params("~/gpt/tests/wilson.txt")
# print(pf)

# take slow reference implementation of wilson (kappa = 1/2/(m0 + 4) ) ;
w_ref = g.qcd.fermion.reference.wilson(U, p)

# and fast Grid version
w = g.qcd.fermion.wilson_clover(U, p, kappa=0.137)

# create point source
src = rng.cnormal(g.vspincolor(grid))

dst_ref, dst = g.lattice(src), g.lattice(src)

# correctness
dst_ref @= w_ref * src
dst @= w * src

eps = g.norm2(dst - dst_ref) / g.norm2(dst)
g.message("Test wilson versus reference:", eps)
assert eps < 1e-13

# now timing
t0 = g.time()
for i in range(100):
    w_ref(dst_ref, src)
t1 = g.time()
for i in range(100):
    w(dst, src)
t2 = g.time()
for i in range(100):
    dst = w(src)
t3 = g.time()
for i in range(100):
    dst @= w * src
t4 = g.time()

g.message("Reference time/s: ", t1 - t0)
g.message("Grid time/s (reuse lattices): ", t2 - t1)
g.message("Grid time/s (with temporaries): ", t3 - t2)
g.message("Grid time/s (with expressions): ", t4 - t3)

# create point source
src = g.mspincolor(grid)
g.create.point(
    src, [1, 0, 0, 0]
)  # pick point 1 so that "S" in preconditioner contributes to test

# build solver using g5m and cg
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-6, "maxiter": 1000})

slv = w.propagator(inv.preconditioned(pc.g5m_ne(), cg))
slv_eo1 = w.propagator(inv.preconditioned(pc.eo1_ne(), cg))
slv_eo2 = w.propagator(inv.preconditioned(pc.eo2_ne(), cg))

# propagator
dst_eo1 = g.mspincolor(grid)
dst_eo2 = g.mspincolor(grid)

dst_eo1 @= slv_eo1 * src
iter_eo1 = len(cg.history)

dst_eo2 @= slv_eo2 * src
iter_eo2 = len(cg.history)

eps2 = g.norm2(dst_eo1 - dst_eo2) / g.norm2(dst_eo1)
g.message(
    f"Result of test EO1 versus EO2 preconditioning: eps2={eps2} iter1={iter_eo1} iter2={iter_eo2}"
)
assert eps2 < 1e-12

# true residuum
eps2 = g.norm2(w * dst_eo1 - src) / g.norm2(src)
g.message("Result of M M^-1 = 1 test: eps2=", eps2)
assert eps2 < 1e-10

# and a reference
if True:
    dst = g.mspincolor(grid)
    dst @= slv * src
    eps2 = g.norm2(dst_eo1 - dst) / g.norm2(dst_eo1)
    g.message("Result of test EO1 versus G5M: eps2=", eps2)
    assert eps2 < 1e-10

dst = dst_eo2

# two-point
correlator = g.slice(g.trace(dst * g.adj(dst)), 3)

# test value of correlator
correlator_ref = [
    1.0710210800170898,
    0.08988216519355774,
    0.015699388459324837,
    0.003721018321812153,
    0.0010877142194658518,
    0.0003579717595130205,
    0.00012700144725386053,
    5.180457083042711e-05,
    3.406393443583511e-05,
    5.2738148951902986e-05,
    0.0001297977869398892,
    0.0003634534077718854,
    0.0011047901352867484,
    0.0037904218770563602,
    0.015902264043688774,
    0.09077762067317963,
]

# output
for t, c in enumerate(correlator):
    g.message(t, c.real, correlator_ref[t])

eps = np.linalg.norm(np.array(correlator) - np.array(correlator_ref))
g.message("Expected correlator eps: ", eps)
assert eps < 1e-5


# split grid solver check
slv_split_eo1 = w.propagator(
    inv.preconditioned(
        pc.eo1_ne(), inv.split(cg, mpi_split=g.default.get_ivec("--mpi_split", None, 4))
    )
)
dst_split = g.mspincolor(grid)
dst_split @= slv_split_eo1 * src
eps2 = g.norm2(dst_split - dst_eo1) / g.norm2(dst_eo1)
g.message(f"Split grid solver check {eps2}")
assert eps2 < 1e-12


# gauge transformation check
V = rng.element(g.mcolor(grid))
prop_on_transformed_U = w.updated(g.qcd.gauge.transformed(U, V)).propagator(
    inv.preconditioned(pc.eo2_ne(), cg)
)
prop_transformed = g.qcd.gauge.transformed(slv_eo2, V)
src = rng.cnormal(g.vspincolor(grid))
dst1 = g(prop_on_transformed_U * src)
dst2 = g(prop_transformed * src)
eps2 = g.norm2(dst1 - dst2) / g.norm2(dst1)
g.message(f"Gauge transformation check {eps2}")
assert eps2 < 1e-12


# test instantiation of other actions
rhq = g.qcd.fermion.rhq_columbia(
    U, mass=4.0, cp=3.0, zeta=2.5, boundary_phases=[1, 1, 1, -1]
)
