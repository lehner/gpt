#!/usr/bin/env python3
#
# Authors: Christoph Lehner, Mattia Bruno, Gabriele Morandi 2023
#
# HMC for 2D CP^N-1 theory
#
import gpt as g
import sys, os
import numpy

g.default.set_verbose("step_size", False)
grid = g.grid([42, 42], g.double)
rng = g.random("hmc-cpn-model")

# action for conj. momenta:
g.message(f"Lattice = {grid.fdimensions}")
g.message("Actions:")
a0 = g.qcd.scalar.action.mass_term()
g.message(f" - {a0.__name__}")

# CP^N-1 action
beta = 0.70
N = 10
a1 = g.qcd.scalar.action.cpn(N, beta)
g.message(f" - {a1.__name__}")

# fields
z = g.vcomplex(grid, N)
a1.draw(z, rng)
l = [g.u1(grid) for _ in range(grid.nd)] 
rng.element(l)
fields = [z] + l

# conjugate momenta
mom_z = g.group.cartesian(z)
mom_l = g.group.cartesian(l)
moms = [mom_z] + mom_l

def hamiltonian():
    return a0(mom_z) + a0(mom_l) + a1(fields)


# molecular dynamics
sympl = g.algorithms.integrator.symplectic

ip_z = sympl.update_p(mom_z, lambda: a1.gradient(fields, z))
ip_l = sympl.update_p(mom_l, lambda: a1.gradient(fields, l))

class constrained_iq(sympl.symplectic_base):
    def __init__(self, fields, moms):
        z = fields[0]
        l = fields[1:]
        mom_z = moms[0]
        mom_l = moms[1:]

        iq_l = sympl.update_q(l, lambda: a0.gradient(mom_l, mom_l))

        def inner(eps):
            a1.constrained_leap_frog(eps, z, mom_z)
            iq_l(eps)
        super().__init__(1, [], [inner], None, "constrained iq")
        
iq = constrained_iq(fields, moms)

# integrator
mdint = sympl.leap_frog(50, [ip_z, ip_l], iq)
g.message(f"Integration scheme:\n{mdint}")

# metropolis
metro = g.algorithms.markov.metropolis(rng)

# MD units
tau = 1.0
g.message(f"tau = {tau} MD units")


def hmc(tau, moms):
    mom_z = moms[0]
    mom_l = moms[1:]
    
    rng.normal_element(mom_l)
    a1.draw(mom_z, rng, z)
    
    accrej = metro(fields)
    h0 = hamiltonian()
    mdint(tau)
    h1 = hamiltonian()
    return [accrej(h1, h0), h1 - h0]


# thermalization
for i in range(1, 21):
    h = []
    for _ in range(10):
        h += [hmc(tau, moms)]
    h = numpy.array(h)
    g.message(f"{i*5} % of thermalization completed")
    g.message(
        f"Action = {a1(fields)}, Acceptance = {numpy.mean(h[:,0]):.2f}, |dH| = {numpy.mean(numpy.abs(h[:,1])):.4e}"
    )

# measure action
def measure():
    return [a1(fields) / (N * beta * grid.fsites * grid.nd)]


# production
history = []
data = []
for i in range(100):
    for k in range(10):
        history += [hmc(tau, moms)]
    data += [measure()]
    g.message(f"Trajectory {i}")

history = numpy.array(history)
g.message(f"Acceptance rate = {numpy.mean(history[:,0]):.2f}")
g.message(f"<|dH|> = {numpy.mean(numpy.abs(history[:,1])):.4e}")

data = numpy.array(data)
g.message(f"Energy density   = {numpy.mean(data[:,0])}")
