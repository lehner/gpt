#!/usr/bin/env python3
#
# Authors: Christoph Lehner, Mattia Bruno 2021
#
# HMC for phi^4 scalar theory
#
import gpt as g
import sys, os
import numpy

beta = g.default.get_float("--beta", 5.96)

g.default.set_verbose("omf4")

grid = g.grid([8, 8, 8, 16], g.double)
rng = g.random("hmc-pure-gauge")

U = g.qcd.gauge.unit(grid)
rng.normal_element(U)

# conjugate momenta
mom = g.group.cartesian(U)

# Log
g.message(f"Lattice = {grid.fdimensions}")
g.message("Actions:")
# action for conj. momenta
a0 = g.qcd.scalar.action.mass_term()
g.message(f" - {a0.__name__}")

# wilson action
a1 = g.qcd.gauge.action.wilson(beta)
g.message(f" - {a1.__name__}")


def hamiltonian():
    return a0(mom) + a1(U)


# molecular dynamics
sympl = g.algorithms.integrator.symplectic

ip = sympl.update_p(mom, lambda: a1.gradient(U, U))
iq = sympl.update_q(U, lambda: a0.gradient(mom, mom))

# integrator
mdint = sympl.OMF4(5, ip, iq)
g.message(f"Integration scheme:\n{mdint}")

# metropolis
metro = g.algorithms.markov.metropolis(rng)

# MD units
tau = 2.0
g.message(f"tau = {tau} MD units")


def hmc(tau, mom):
    rng.normal_element(mom)
    accrej = metro(U)
    h0 = hamiltonian()
    mdint(tau)
    h1 = hamiltonian()
    return [accrej(h1, h0), h1 - h0]


# thermalization
ntherm = 100
for i in range(1, 11):
    h = []
    timer = g.timer("hmc")
    for _ in range(ntherm // 10):
        timer("trajectory")
        h += [hmc(tau, mom)]
    h = numpy.array(h)
    timer()
    g.message(f"{i*10} % of thermalization completed")
    g.message(
        f'Average time per trajectory = {timer.dt["trajectory"]/ntherm*10:g} secs'
    )
    g.message(
        f"Plaquette = {g.qcd.gauge.plaquette(U)}, Acceptance = {numpy.mean(h[:,0]):.2f}, |dH| = {numpy.mean(numpy.abs(h[:,1])):.4e}"
    )

# production
history = []
data = []
n = 100
dtrj = 10
for i in range(n):
    for k in range(dtrj):
        history += [hmc(tau, mom)]
    data += [g.qcd.gauge.plaquette(U)]
    g.message(f"Trajectory {i}, {history[-1]}")

history = numpy.array(history)
g.message(f"Acceptance rate = {numpy.mean(history[:,0]):.2f}")
g.message(f"<|dH|> = {numpy.mean(numpy.abs(history[:,1])):.4e}")

data = numpy.array(data)
g.message(f"<plaq>   = {numpy.mean(data[:,0])}")
