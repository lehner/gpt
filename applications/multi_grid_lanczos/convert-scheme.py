#!/usr/bin/env python3
import gpt as g
import numpy as np
import sys

conf = g.default.get_single("--conf", None)
g.message(f"Fixing {conf}")

evec_in = (
    "/gpfs/alpine/phy138/proj-shared/phy138flavor/lehner/runs/summit-96I-"
    + conf
    + "-256/lanczos.output"
)
evec_out = (
    "/gpfs/alpine/phy138/proj-shared/phy138flavor/lehner/runs/summit-96I-"
    + conf
    + "-256/lanczos.output.fixed"
)
fmt = g.format.cevec({"nsingle": 100, "max_read_blocks": 16})
U = g.convert(
    g.load(
        "/gpfs/alpine/phy138/proj-shared/phy138flavor/chulwoo/evols/96I2.8Gev/evol0/configurations/ckpoint_lat."
        + conf
    ),
    g.single,
)
eps_norm = 1e-4
eps2_evec = 1e-5
eps_eval = 1e-2
nskip = 1
load_from_alternative_scheme = True

qz = g.qcd.fermion.mobius(
    U,
    {
        "mass": 0.00054,
        "M5": 1.8,
        "b": 1.5,
        "c": 0.5,
        "Ls": 12,
        "boundary_phases": [1.0, 1.0, 1.0, -1.0],
    },
)

# no modification needed below
basis, cevec, fev = g.load(
    evec_in, grids=qz.F_grid_eo, alternative_scheme=load_from_alternative_scheme
)
b = g.block.map(cevec[0].grid, basis)

g.message("Norm test")
for i in range(len(basis)):
    n = g.norm2(basis[i])
    g.message(i, n, cevec[0].grid.gsites)
    assert abs(n / cevec[0].grid.gsites - 1) < eps_norm

# g.message("Before ortho")
# b.orthonormalize()
# g.message("After ortho")
Mpc = g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd)(qz).Mpc
for i in range(0, len(basis), nskip):
    evec = g(b.promote * cevec[i])
    n = g.norm2(evec)
    g.message("Norm evec:", i, n)
    assert abs(n - 1) < eps_norm

    ev, eps2 = g.algorithms.eigen.evals(Mpc, [evec], real=True)[0]
    assert all([e2 < eps2_evec for e2 in eps2])
    g.message(i, ev, fev[i])
    assert abs(ev / fev[i] - 1) < eps_eval

    cevecPrime = g(b.project * evec)
    g.message(
        "Test:",
        g.norm2(cevecPrime - cevec[i]) / g.norm2(cevecPrime),
        g.norm2(cevecPrime),
        g.norm2(cevec[i]),
    )

if load_from_alternative_scheme:
    g.save(evec_out, [basis, cevec, fev], fmt)
