#!/usr/bin/env python3
import gpt as g
import os

hc = g.qcd.honeycomb()

beta = g.default.get_float("--beta", None)
L = g.default.get_int("--L", None)
nsteps = g.default.get_int("--nsteps", 5)
nmax = 1000
grid = g.grid([L] * 4, g.double)

U = [g.mcolor(grid) for _ in range(hc.number_of_link_fields)]
mom = g.group.cartesian(U)


rng = g.random("test")

rng.normal_element(U, scale=0.3)

a0 = g.qcd.scalar.action.mass_term()
a1 = hc.gauge.action.wilson(beta, grid)


def hamiltonian():
    return a0(mom) + a1(U)


sympl = g.algorithms.integrator.symplectic

ip = sympl.update_p(mom, lambda: a1.gradient(U, U))
iq = sympl.update_q(U, lambda: a0.gradient(mom, mom))

# integrator
mdint = sympl.OMF4(nsteps, ip, iq)


def hmc(tau):
    rng.normal_element(mom)
    h0 = hamiltonian()
    mdint(tau)
    h1 = hamiltonian()
    return h1 - h0


root = f"hc.{beta}.{L}"
if g.rank() == 0:
    os.makedirs(root, exist_ok=True)

latest = None
for i in range(nmax):
    if os.path.exists(f"{root}/ckpoint_lat.{i}"):
        latest = i

if latest is not None:
    rng = g.random(f"{root}/ckpoint_lat.{latest}")
    g.copy(U, g.load(f"{root}/ckpoint_lat.{latest}"))
else:
    latest = 0

for i in range(latest + 1, nmax):
    dH = hmc(1.0)
    P = hc.gauge.plaquette(U)
    Phc = [g.qcd.gauge.plaquette(hc.hypercube(j, U)) for j in range(2)]
    g.message(i, dH, P, Phc)
    f = open(f"{root}/ckpoint_lat.{i}.log", "wt")
    f.write(f"dH {dH}\n")
    f.write(f"P {P}\n")
    for j in range(2):
        f.write(f"Phc{j} {Phc[j]}\n")
    f.close()

    if i % 10 == 0:
        for j in range(2):
            Uhc = hc.hypercube(j, U)
            step = 0.05
            tau = 0
            f = open(f"{root}/ckpoint_lat.{i}.E{j}", "wt")
            for n in range(80):
                tau += step
                g.message(tau)
                Uhc = g.qcd.gauge.smear.wilson_flow(Uhc, step)
                E = g.qcd.gauge.energy_density(Uhc).real
                f.write(f"{tau} {E}\n")
            f.close()

        g.save(f"{root}/ckpoint_lat.{i}", U)

g.barrier()
