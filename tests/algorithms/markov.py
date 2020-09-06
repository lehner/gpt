#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020, Mattia Bruno 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt
import numpy

gpt.default.set_verbose("hmc")
grid = gpt.grid([16, 16, 16, 16], gpt.double)

rng = gpt.random("test")

phi = gpt.complex(grid)
rng.normal(phi, sigma=0.5)
phi[:].imag = 0

mom = gpt.algorithms.markov.conjugate_momenta(phi)

a0 = gpt.qcd.actions.scalar.phi4(phi, 0.25, 0.0)

iphi = gpt.algorithms.integrators.update_scalar(phi, mom)
i0 = gpt.algorithms.integrators.update_mom(mom, a0)

lp = gpt.algorithms.integrators.OMF2(4, i0, iphi)

hmc = gpt.algorithms.markov.hmc(phi, mom, None, lp, rng)

hmc(2.0)

assert hmc.reversibility_test(0.1) < 1e-12
