#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import gpt.qcd.quarkContract as qC
import time

# load configuration
rng = g.random("test")
L, Nt = 4, 8
U0 = g.qcd.gauge.random(g.grid([L, L, L, Nt], g.double), rng)
grid = U0[0].grid
Vol = np.array(grid.fdimensions) # eg [24, 24, 24, 48]
rng = g.random("test")

# apply gauge transformation to the links
V = rng.lie(g.lattice(U0[0]))
U_transformed = g.qcd.gauge.transformed(U0, V)

gauge_links = [U0, U_transformed]
g.message(type(gauge_links))


# create point source
g.message("Creating the point source")
src = g.mspincolor(grid)
g.create.point(src, [0, 0, 0, 4])

correlators = np.empty((2, 6, 2, Nt), dtype=float)

for n, U in enumerate(gauge_links):

    # quark
    w = g.qcd.fermion.wilson_clover(U,{
        "kappa" : 0.137,
        "csw_r" : 0.,
        "csw_t" : 0.,
        "xi_0" : 1,
        "nu" : 1,
        "isAnisotropic" : False,
        "boundary_phases" : [ 1.0, 1.0, 1.0, -1.0 ]
    })
    # build solver using g5m and cg
    inv = g.algorithms.inverter
    pc = g.qcd.fermion.preconditioner
    cg = inv.cg({"eps": 1e-15, "maxiter": 1000})
    slv_eo2 = w.propagator(inv.preconditioned(pc.eo2_ne(), cg))

    # propagator
    dst = g.mspincolor(grid)
    dst @= slv_eo2 * src

    propagators = [dst, g.eval(g.gamma[5] * g.adj(dst) * g.gamma[5])]

    for l, prop in enumerate(propagators):
        contraction_routines = [qC.quarkContract12(prop, prop),
                                qC.quarkContract13(prop, prop),
                                qC.quarkContract14(prop, prop),
                                qC.quarkContract23(prop, prop),
                                qC.quarkContract24(prop, prop),
                                qC.quarkContract34(prop, prop)]

        for m, di_quark in enumerate(contraction_routines):
            correlator = g.slice(g.trace(di_quark * g.adj(di_quark) ), 3)

            for t, c in enumerate(correlator):
                correlators[n, m, l, t] = c.real

# test gauge covariance
for k in range(len(contraction_routines)):
    for t in range(Nt):
        assert (correlators[0, k, l, t] - correlators[1, k, l, t] < 1e-12)

# test spin structure
for n in range(4):
    I = g.ot_matrix_spin_color(4, 3).identity()
    if (n == 0):
        di_quark1 = qC.quarkContract12(dst, dst)
        di_quark2 = qC.quarkContract12(dst, g.eval(I * g.spin_trace(dst)))

        correlator1 = g.slice(g.trace(di_quark1), 3)
        correlator2 = g.slice(g.color_trace(di_quark2), 3)

        for t in range(Nt):
            assert (correlator1[t] - correlator2[t][0, 0]) < 1e-13

    if (n == 1):
        di_quark1 = qC.quarkContract34(dst, dst)
        di_quark2 = qC.quarkContract34(g.eval(I * g.spin_trace(dst)), dst)

        correlator1 = g.slice(g.trace(di_quark1), 3)
        correlator2 = g.slice(g.color_trace(di_quark2), 3)

        for t in range(Nt):
            assert (correlator1[t] - correlator2[t][0, 0]) < 1e-13

    if (n == 2):
        di_quark1 = qC.quarkContract13(dst, dst)
        di_quark2 = qC.quarkContract24(dst, dst)

        correlator1 = g.slice(g.trace(di_quark1), 3)
        correlator2 = g.slice(g.trace(di_quark2), 3)

        for t in range(Nt):
            assert (correlator1[t].real - correlator2[t].real) < 1e-13

    if (n == 3):
        di_quark1 = qC.quarkContract14(dst, dst)
        di_quark2 = qC.quarkContract23(dst, dst)
        correlator1 = g.slice(g.trace(di_quark1), 3)
        correlator2 = g.slice(g.trace(di_quark2), 3)

        for t in range(Nt):
            assert (correlator1[t].real - correlator2[t].real) < 1e-13
