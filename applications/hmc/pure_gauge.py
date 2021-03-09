#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020, Mattia Bruno 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt
import numpy

gpt.default.set_verbose("hmc")
gpt.default.set_verbose("omf4")
gpt.default.set_verbose("random", False)
grid = gpt.grid([16, 16, 32, 32], gpt.double)

rng = gpt.random("pure-gauge")
U = gpt.qcd.gauge.unit(grid)

mom = gpt.algorithms.markov.conjugate_momenta(U)

act = gpt.qcd.actions.gauge.wilson(U, 5.96)
act.hot_start(rng)

N=10

time = gpt.timer("force")
for i in range(N):
    time('setup')
    act.setup_force()
time()
gpt.message('time per force',time.total/N)

import sys
sys.exit()

int = gpt.algorithms.integrators
mdint = int.OMF4(8, int.update_mom(mom, act), int.update_gauge(U, mom))

hmc = gpt.algorithms.markov.hmc(U, mom, None, mdint, rng)

gpt.message("Thermalization")
for itrj in range(1): #40
    gpt.message(f"Trajectory {itrj}")
    hmc(2.0)
    gpt.message(f"Plaquette = {gpt.qcd.gauge.plaquette(U)*3.}")

hmc_hist = []
plaq_hist = []
for itrj in range(1): #100
    gpt.message(f"Trajectory {itrj}")
    hmc_hist.append(hmc(2.0))
    plaq_hist.append(gpt.qcd.gauge.plaquette(U))
    gpt.message(f"Plaquette = {plaq_hist[-1]*3.}")

hmc_hist = numpy.array(hmc_hist)
gpt.message(f"Average acceptance rate = {numpy.mean(hmc_hist[:,0]):g}")
gpt.message(f"Average dH = {numpy.mean(hmc_hist[:,1]):g}")

gpt.message(f"Plaquette reference value = {1.767}")
