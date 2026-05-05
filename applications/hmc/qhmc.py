#!/usr/bin/env python3
#
# Authors: Christoph Lehner
#
# Symmetric HMC with left and right updates
#
import gpt as g
import sys, os
import numpy as np

noutersteps = g.default.get_int("--noutersteps", 5)
tau = g.default.get_float("--tau", 1.0)
beta = g.default.get_float("--beta", 2.95)
seed = g.default.get("--seed", "hmc-pure-gauge")
n = g.default.get_int("--n", 1000)
root = g.default.get("--root", None)
implicit_eps = g.default.get_float("--eps", 1e-9)
Q_0 = g.default.get_float("--Q0", None)
Q_1 = g.default.get_float("--Q1", None)
Q_sm = g.default.get_int("--Qsm", None)
Q_c = [float(x) for x in g.default.get("--Qc", None).split(";")]

g.default.set_verbose("omf4")

nsave = 1
if tau < 1.0:
    nsave = int(np.round(1.0 / tau))
    g.message("Save every", nsave," config")

grid = g.grid([32, 32, 32, 48], g.double)
#grid = g.grid([8, 8, 8, 8], g.double)
rng = g.random(seed)

U = g.qcd.gauge.random(grid, rng)
#W = [g.matrix_color_complex_additive(grid, 3) for i in range(4)]
#for i in range(4):
#    g.copy(W[i], U[i])
#del U
    
if g.rank() == 0:
    os.makedirs(root, exist_ok=True)

g.barrier()

        

fn_try = None
i0 = 0
for i in range(0, n):
    fn = f"{root}/ckpoint_lat.{i}" #.W
    if os.path.exists(fn):
        fn_try = fn
        i0 = i + 1

if fn_try is not None:
    U = g.load(fn_try) # W
    rng = g.random(fn_try)

# Log
g.message(f"Lattice = {grid.fdimensions}")
g.message("Actions:")

a0 = g.qcd.scalar.action.mass_term()

# conjugate momenta
p_mom = g.group.cartesian(U)
rng.element(p_mom)

a0.assert_gradient_error(rng, p_mom, p_mom, 1e-3, 1e-8)
v0 = a0.draw(p_mom, rng)
v0b = a0(p_mom)
eps = abs(v0 - v0b) / abs(v0)
assert eps < 1e-13

# wilson action
a1 = g.qcd.gauge.action.iwasaki(beta)

# topology action
aQ = g.qcd.gauge.action.topology(U, Q_0, Q_1, Q_c)
sm = g.qcd.gauge.smear.stout(rho=0.12)
for it in range(Q_sm):
    aQ = aQ.transformed(sm)
#aQ.assert_gradient_error(rng, U, U, 1e-3, 1e-8)

a1 = a1 + aQ

def hamiltonian():
    return a0(p_mom) + a1(U)

# molecular dynamics
sympl = g.algorithms.integrator.symplectic

ip = sympl.update_p(p_mom, lambda: a1.gradient(U, U))
iq = sympl.update_q(U, lambda: a0.gradient(p_mom, p_mom))
integrator = sympl.OMF2(noutersteps, ip, iq)

# integrator
g.message(f"Integration scheme:\n{integrator}")

# metropolis
metro = g.algorithms.markov.metropolis(rng)

# MD units
g.message(f"tau = {tau} MD units")


def hmc():
    a0.draw(p_mom, rng)
    accrej = metro(U)
    h0 = hamiltonian()
    integrator(tau)
    h1 = hamiltonian()
    return [True, h1-h0]
    #return [accrej(h1, h0), h1 - h0]

# production
for i in range(i0, n):
    _, dH = hmc()
    P = g.qcd.gauge.plaquette(U)
    g.message(f"Trajectory {i}, P={P}, dH={dH}")

    if i % nsave != 0:
        continue

    # polyakov
    PL = g.identity(U[3])
    for t in range(grid.gdimensions[3]):
        PL = g(U[3] * g.cshift(PL, 3, 1))
    polyakov = g(g.trace(PL)/3)[:,:,:,0]

    if g.rank() == 0:
        flog = open(f"{root}/ckpoint_lat.{i}.log","wt")
        flog.write(f"dH {dH}\n")
        flog.write(f"P {P}\n")
        for p in polyakov:
            flog.write(f"L {p[0].real} {p[0].imag}\n")
    # and wilson flowed energy
    Uwf = U
    twf = 0.0
    for _ in range(80):
        g.message("Wilson flow", twf)
        Uwf = g.qcd.gauge.smear.wilson_flow(Uwf, epsilon=0.1)
        twf += 0.1
        E = g.qcd.gauge.energy_density(Uwf).real
        Q = g.qcd.gauge.topological_charge(Uwf).real
        if g.rank() == 0:
            flog.write(f"EQ {twf} {E} {Q}\n")
            flog.flush()

    g.message(f"At twf={twf}: E = {E}, Q = {Q}")
    if g.rank() == 0:
        flog.close()

    g.barrier()

    g.save(f"{root}/ckpoint_lat.{i}", U, g.format.nersc())

