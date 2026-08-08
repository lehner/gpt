#!/usr/bin/env python3
import gpt as g

hc = g.qcd.honeycomb()

grid = g.grid([4, 4, 4, 8], g.double)
U = [g.mcolor(grid) for _ in range(hc.number_of_link_fields)]
mom = g.group.cartesian(U)


rng = g.random("test")

rng.normal_element(U, scale=0.3)

a0 = g.qcd.scalar.action.mass_term()
a1 = hc.gauge.action.wilson(26, grid)


def hamiltonian():
    return a0(mom) + a1(U)


sympl = g.algorithms.integrator.symplectic

ip = sympl.update_p(mom, lambda: a1.gradient(U, U))
iq = sympl.update_q(U, lambda: a0.gradient(mom, mom))

# integrator
mdint = sympl.OMF4(5, ip, iq)


def hmc(tau):
    rng.normal_element(mom)
    h0 = hamiltonian()
    mdint(tau)
    h1 = hamiltonian()
    return h1 - h0


for i in range(100):
    g.message("dH", hmc(1.0))
    g.message(hc.gauge.plaquette(U))
