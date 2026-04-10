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
rho = g.default.get_float("--rho", None)
# tmasses = [float(x) for x in g.default.get("--t_masses", None).split(";")]
g.default.set_verbose("omf4")

grid = g.grid([32, 32, 32, 48], g.double)
#grid = g.grid([8, 8, 8, 8], g.double)
rng = g.random(seed)

U = g.qcd.gauge.unit(grid)
rng.normal_element(U, scale=0.1)

if g.rank() == 0:
    os.makedirs(root, exist_ok=True)

g.barrier()


fn_try = None
i0 = 0
for i in range(0, n):
    fn = f"{root}/ckpoint_lat.{i}"
    if os.path.exists(fn):
        fn_try = fn
        i0 = i + 1

if fn_try is not None:
    U0 = g.load(fn_try)
    rng = g.random(fn_try)
    for mu in range(4):
        U[mu] @= U0[mu]

# Log
g.message(f"Lattice = {grid.fdimensions}")
g.message("Actions:")
# action for conj. momenta
even, odd = g.even_odd_projectors(U[0].grid)

full = g(even + odd)
none = g(0 * full)

description = [
    [(rho, g.path().f(nu).f(mu).b(nu, 2).b(mu).f(nu)) for nu in range(4) if mu != nu]
    for mu in range(4)
]

pt_e = [
    g.qcd.gauge.smear.parallel_transport(
        U,
        description,
        [even if i == j else full for i in range(4)],
        [odd if i == j else none for i in range(4)],
    )
    for j in range(4)
]

pt_o = [
    g.qcd.gauge.smear.parallel_transport(
        U,
        description,
        [odd if i == j else full for i in range(4)],
        [even if i == j else none for i in range(4)],
    )
    for j in range(4)
]

inv_pt_e = [x.inv() for x in pt_e]
inv_pt_o = [x.inv() for x in pt_o]

for j in range(4):
    test_U = inv_pt_e[j](pt_e[j](U))
    for i in range(4):
        eps = g.norm2(test_U[i] - U[i]) / g.norm2(U[i])
        g.message(f"Test inversion (e{j}{i}): {eps}")
        assert eps < 1e-25
    test_U = inv_pt_o[j](pt_o[j](U))
    for i in range(4):
        eps = g.norm2(test_U[i] - U[i]) / g.norm2(U[i])
        g.message(f"Test inversion (e{j}{i}): {eps}")
        assert eps < 1e-25

a1 = g.qcd.gauge.action.iwasaki(beta)
# undo smearing
for i in reversed(range(4)):
    U = inv_pt_e[i](U)
    U = inv_pt_o[i](U)


# g.message("Middle", g.qcd.gauge.plaquette(U), a1(U))

a0 = g.qcd.scalar.action.mass_term()

# conjugate momenta
p_mom = g.group.cartesian(U)

# wilson action
a1 = g.qcd.gauge.action.iwasaki(beta)
for i in reversed(range(4)):
    a1 = a1.transformed(pt_e[i])
    a1 = a1.transformed(pt_o[i])

a1.assert_gradient_error(rng, U, U, 1e-4, 1e-7)


def hamiltonian():
    return a0(p_mom) + a1(U)


# molecular dynamics
sympl = g.algorithms.integrator.symplectic

ip = sympl.update_p(p_mom, lambda: a1.gradient(U, U))
iq = sympl.update_q(U, lambda: a0.gradient(p_mom, p_mom))
integrator = sympl.OMF4(noutersteps, ip, iq)

# integrator
g.message(f"Integration scheme:\n{integrator}")

# test error scaling of integrator
h0 = hamiltonian()


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
    return [True, h1 - h0]
    # return [accrej(h1, h0), h1 - h0]


# production
for i in range(i0, n):
    _, dH = hmc()

    Usm = U
    for ii in range(4):
        Usm = pt_o[ii](Usm)
        Usm = pt_e[ii](Usm)

    P = g.qcd.gauge.plaquette(Usm)
    g.message(f"Trajectory {i}, P={P} (integration {g.qcd.gauge.plaquette(U)}), dH={dH}")
    for xx in [1, 2, 3, 4]:
        g.message(
            f"{xx} x 1 rectangle for physical field {g.qcd.gauge.rectangle(Usm, xx, 1)} versus integration variable {g.qcd.gauge.rectangle(U, xx, 1)}"
        )

    # polyakov
    PL = g.identity(Usm[3])
    for t in range(grid.gdimensions[3]):
        PL = g(Usm[3] * g.cshift(PL, 3, 1))
    polyakov = g(g.trace(PL) / 3)[:, :, :, 0]

    if g.rank() == 0:
        flog = open(f"{root}/ckpoint_lat.{i}.log", "wt")
        flog.write(f"dH {dH}\n")
        flog.write(f"P {P}\n")
        for p in polyakov:
            flog.write(f"L {p[0].real} {p[0].imag}\n")
    # and wilson flowed energy
    Uwf = Usm
    twf = 0.0
    for _ in range(80):
        Uwf = g.qcd.gauge.smear.wilson_flow(Uwf, epsilon=0.1)
        twf += 0.1
        E = g.qcd.gauge.energy_density(Uwf).real
        Q = g.qcd.gauge.topological_charge(Uwf).real
        if g.rank() == 0:
            flog.write(f"EQ {twf} {E} {Q}\n")

    g.message(f"At twf={twf}: E = {E}, Q = {Q}")
    if g.rank() == 0:
        flog.close()

    g.barrier()

    g.save(f"{root}/ckpoint_lat.{i}", Usm, g.format.nersc())
