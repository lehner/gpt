#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020, Mattia Bruno 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt
import numpy

gpt.default.set_verbose("hmc")
grid = gpt.grid([16, 16], gpt.double)

rng = gpt.random("test")

phi = gpt.complex(grid)
rng.normal(phi, sigma=0.5)
phi[:].imag = 0

mom = gpt.algorithms.markov.conjugate_momenta(phi)

a0 = gpt.qcd.scalar.actions.phi4(phi, 0.25, 0.0)

iphi = gpt.algorithms.integrators.update(phi, mom.mom)
i0 = gpt.algorithms.integrators.update(mom.mom, a0)

lp = gpt.algorithms.integrators.leap_frog(20, i0, iphi)

hmc = gpt.algorithms.markov.hmc(phi, mom, None, lp, rng)

# thermalization
for itrj in range(100):
    hmc(2.0)


hmc_hist = []
act = []

for itrj in range(100):
    gpt.message(f"Trajectory {itrj}")
    hmc_hist.append(hmc(2.0))
    act.append(a0())

hmc_hist = numpy.array(hmc_hist)

gpt.message(f"Average acceptance rate = {numpy.mean(hmc_hist[:,0]):g}")
gpt.message(f"Average dH = {numpy.mean(hmc_hist[:,1]):g}")
gpt.message(
    f"<Action>/Analytic = {numpy.mean(act)/grid.fsites*2:0.4f} +- {numpy.std(act)/grid.fsites*2:0.4f}"
)

assert hmc.reversibility_test(0.1) < 1e-12
