#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys

# gauge field
rng = g.random("test")
U = g.qcd.gauge.random(g.grid([16, 8, 8, 16], g.double), rng)

# zmobius operator
q = g.qcd.fermion.zmobius(
    U,
    {
        "mass": 0.08,
        "M5": 1.8,
        "b": 1.0,
        "c": 0.0,
        "omega": [
            0.07479396293100343 + 1j * (-0.07180640088469024),
            0.11348176576169644 + 1j * (0.01959818142922749),
            0.17207275433484948 + 1j * (0),
            0.5906659797999617 + 1j * (0),
            1.2127961700946597 + 1j * (0),
            1.8530255403075104 + 1j * (0),
            1.593983619964445 + 1j * (0),
            0.8619627395089003 + 1j * (0),
            0.39706365411914263 + 1j * (0),
            0.26344003875987015 + 1j * (0),
            0.11348176576169644 + 1j * (-0.01959818142922749),
            0.07479396293100343 + 1j * (0.07180640088469024),
        ],
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    },
)

# shortcuts
pc = g.qcd.fermion.preconditioner

# random vectors on F_grid_eo
N = 10
evec = [g.vspincolor(q.F_grid_eo) for i in range(N)]
evals = [rng.normal() for i in range(N)]
rng.cnormal(evec)

# test source
F_src = g.vspincolor(q.F_grid)
rng.cnormal(F_src)

U_src = g.vspincolor(q.U_grid)
rng.cnormal(U_src)

# temporaries
ie, io, oe, oo = tuple([g.vspincolor(q.F_grid_eo) for i in range(4)])

# physical import/export
exp = q.ExportPhysicalFermionSolution
imp = q.ImportPhysicalFermionSource

for p, tag, parity in [
    (pc.eo1_ne(parity=g.odd), "eo1-odd", g.odd),
    (pc.eo2_ne(parity=g.odd), "eo2-odd", g.odd),
    (pc.eo1_ne(parity=g.even), "eo1-even", g.even),
    (pc.eo2_ne(parity=g.even), "eo2-even", g.even),
]:
    g.message("Test", tag)

    for x in evec:
        x.checkerboard(parity)

    p_q = p(q)
    a2a_p = g.algorithms.modes.a2a(pc.physical(p)(q))
    a2a_u = g.algorithms.modes.a2a(p_q)
    lma_unphysical = g.algorithms.inverter.preconditioned(
        p, g.algorithms.inverter.deflate(evec, evals)
    )
    lma_physical = q.propagator(lma_unphysical)

    #########
    # unphysical test (5d for domain wall)
    dst_lma = g.eval(lma_unphysical(q) * F_src)

    # reconstruct by hand
    a2a_unphysical = g.algorithms.modes.matrix(
        [a2a_u.v(x) for x in evec],
        [a2a_u.w(x) for x in evec],
        evals,
        lambda x: 1.0 / x,
    )
    dst_a2a = g.eval(a2a_unphysical * F_src)

    # add the contact term
    S_dst = g.eval(p_q.S * F_src)
    dst_a2a += S_dst

    eps2 = g.norm2(dst_lma - dst_a2a) / g.norm2(dst_lma)
    g.message(
        "Test 5d",
        eps2,
        "with contact term contribution of size",
        g.norm2(S_dst) / g.norm2(dst_lma),
    )
    assert eps2 < 1e-25

    #########
    # physical test
    dst_lma = g.eval(lma_physical * U_src)

    # reconstruct by hand
    a2a_physical = g.algorithms.modes.matrix(
        [a2a_p.v(x) for x in evec], [a2a_p.w(x) for x in evec], evals, lambda x: 1.0 / x
    )
    dst_a2a = g.eval(a2a_physical * U_src)

    # add the contact term
    S_dst = g.eval(exp * p_q.S * imp * U_src)
    dst_a2a += S_dst

    eps2 = g.norm2(dst_lma - dst_a2a) / g.norm2(dst_lma)
    g.message(
        "Test 4d",
        eps2,
        "with contact term contribution of size",
        g.norm2(S_dst) / g.norm2(dst_lma),
    )
    assert eps2 < 1e-25

# v and w (and unphysical versions) are tested at this point
# in IRL test, we have actual eigenvectors of wilson
# and should add a test of long-distance agreement
