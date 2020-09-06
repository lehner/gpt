#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020, Mattia Bruno 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt
import numpy

gpt.default.set_verbose("hmc")
gpt.default.set_verbose("leap_frog")
gpt.default.set_verbose("omf2")
gpt.default.set_verbose("omf4")
gpt.default.set_verbose("random", False)
grid = gpt.grid([16, 16, 16, 16], gpt.double)

rng = gpt.random("test")

phi = gpt.complex(grid)
rng.normal(phi, sigma=0.5)
phi[:].imag = 0

mom = gpt.algorithms.markov.conjugate_momenta(phi)
act = gpt.qcd.actions.scalar.phi4(phi, 0.25, 0.01)

iphi = gpt.algorithms.integrators.update_scalar(phi, mom)
i0 = gpt.algorithms.integrators.update_mom(mom, act)

n = 1

integrators = []
integrators.append(gpt.algorithms.integrators.leap_frog(n, i0, iphi))
integrators.append(gpt.algorithms.integrators.OMF2(n, i0, iphi))
integrators.append(gpt.algorithms.integrators.OMF4(n, i0, iphi))

phi_copy = gpt.copy(phi)
mom_copy = gpt.lattice(phi)
a0 = act()

mom.refresh(rng)
mom_copy @= mom.mom[0]

for eps in [0.1, 0.01, 0.001, 0.0001]:
    a = []
    for i in integrators:
        mom.mom[0] @= mom_copy
        phi @= phi_copy
        i(eps)
        a.append(act())
    da = numpy.fabs(numpy.array(a[0:-1]) - a[-1]) / a[-1]
    gpt.message(f"eps = {eps/n:.2e}, dH = {da[0]:.2e}, {da[1]:.2e}")
