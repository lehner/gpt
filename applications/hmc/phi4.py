#!/usr/bin/env python3
#
# Authors: Christoph Lehner, Mattia Bruno 2021
#
# HMC for phi^4 scalar theory
#
import gpt as g
import sys, os
import numpy

grid = g.grid([16, 16, 16, 16], g.double)
rng = g.random("hmc-phi4")

phi = g.real(grid)
rng.element(phi, scale=0.2)

# conjugate momenta
mom = g.group.cartesian(phi)

# action for conj. momenta
g.message(f"Lattice = {grid.fdimensions}")
g.message("Actions:")
a0 = g.qcd.scalar.action.mass_term()
g.message(f" - {a0.__name__}")

# phi^4 action
kappa = 0.1119
l = 0.01234
a1 = g.qcd.scalar.action.phi4(kappa, l)
g.message(f" - {a1.__name__}")
g.message(f"phi4 mass = {a1.kappa_to_mass(kappa, l, grid.nd)}")


def hamiltonian():
    return a0(mom) + a1(phi)


# molecular dynamics
sympl = g.algorithms.integrator.symplectic

ip = sympl.update_p(mom, lambda: a1.gradient(phi, phi))
iq = sympl.update_q(phi, lambda: a0.gradient(mom, mom))

# integrator
mdint = sympl.OMF2(6, ip, iq)
g.message(f"Integration scheme:\n{mdint}")

# metropolis
metro = g.algorithms.markov.metropolis(rng)

# MD units
tau = 2.0
g.message(f"tau = {tau} MD units")


def hmc(tau, mom):
    rng.normal_element(mom)
    accrej = metro(phi)
    h0 = hamiltonian()
    mdint(tau)
    h1 = hamiltonian()
    return [accrej(h1, h0), h1 - h0]


# thermalization
for i in range(1, 21):
    h = []
    for _ in range(100):
        h += [hmc(tau, mom)]
    h = numpy.array(h)
    g.message(f"{i*5} % of thermalization completed")
    g.message(
        f"Action = {a1(phi)}, Acceptance = {numpy.mean(h[:,0]):.2f}, |dH| = {numpy.mean(numpy.abs(h[:,1])):.4e}"
    )


# measure <phi>, <phi^2>
def measure(phi):
    return [g.sum(phi).real / grid.fsites, g.norm2(phi) / grid.fsites]


# production
history = []
data = []
for i in range(100):
    for k in range(10):
        history += [hmc(tau, mom)]
    data += [measure(phi)]
    g.message(f"Trajectory {i}")

history = numpy.array(history)
g.message(f"Acceptance rate = {numpy.mean(history[:,0]):.2f}")
g.message(f"<|dH|> = {numpy.mean(numpy.abs(history[:,1])):.4e}")

data = numpy.array(data)
g.message(f"<phi>   = {numpy.mean(data[:,0])}")
g.message(f"<phi^2> = {numpy.mean(data[:,1])}")
