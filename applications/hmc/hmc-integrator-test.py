#!/usr/bin/env python3
#
# Authors: Christoph Lehner
#
import gpt as g
import numpy as np
import sys, os

# parameters
tau = g.default.get_float("--tau", 1.0)
root = g.default.get("--root",None)
beta = g.default.get_float("--beta", 2.95)
seed = g.default.get("--seed", "hmc-pure-gauge")
n = g.default.get_int("--n", 1000)

# grid
grid = g.grid([4, 4, 4, 8], g.double)
rng = g.random(seed)

# load state / initialize state
U = g.qcd.gauge.unit(grid)
rng.normal_element(U, scale=0.3)

fn_try = None
i0 = 0
for i in range(0, n):
    fn = f"{root}/ckpoint_lat.{i}"
    if os.path.exists(fn):
        fn_try = fn
        i0 = i + 1

if fn_try is not None:
    rng = g.random(fn_try)
    U0 = g.load(fn_try)
    for mu in range(4):
        U[mu] @= U0[mu]

p_mom = g.group.cartesian(U) # conjugate momenta
aP = g.qcd.scalar.action.mass_term()

# wilson action
aQ = g.qcd.gauge.action.iwasaki(beta)

# test integration
aP.draw(p_mom, rng)

sympl = g.algorithms.integrator.symplectic

def hamiltonian():
    a_gauge = aQ(U)
    a_p_mom = aP(p_mom)
    return a_gauge + a_p_mom

# create OMF2 and OMF2_force_gradient integrators
p_mom2 = g.copy(p_mom)
U2 = g.copy(U)
iq = sympl.update_q(U, lambda: aP.gradient(p_mom, p_mom), "Q")
ip = sympl.update_p(p_mom, lambda: aQ.gradient(U, U), "P")
ip_fg = sympl.update_p_force_gradient(U, iq, p_mom, ip, ip, "P_FG")

integrators = [
    sympl.leap_frog(1, ip, iq),
    sympl.OMF2(1, ip, iq),
    sympl.OMF2(1, ip, iq, 1.0/6.0),
    sympl.OMF2(1, ip, iq, 1.0/6.0, 1.0/6.0),
    sympl.OMF4(1, ip, iq),
    sympl.OMF2_force_gradient(1, ip, iq, ip_fg, 0.18,0.5),
    sympl.OMF2_force_gradient(1, ip, iq, ip_fg)
]

for integrator in integrators:
    g.message(integrator)

# test error scaling of integrator
h0 = hamiltonian()

U0 = g.copy(U)
p_mom0 = g.copy(p_mom) 
for dt in [1.0/8.0,1.0/16.0,1.0/32.,1.0/64.,1.0/128.]:
    for integrator in integrators:
        g.copy(U, U0)
        g.copy(p_mom, p_mom0)
        integrator(dt)
        h1 = hamiltonian()
        integrator(-dt)
        eps2 = sum([g.norm2(u - u2) for u, u2 in zip(U + p_mom, U0 + p_mom0)])
        g.message(f"dH dt={dt} {integrator.__name__} dH={h1-h0:.2g} reversibility={eps2**0.5:.2g}")

