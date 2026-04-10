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
tscale = g.default.get_float("--t_scale", None)
tmasses = [float(x) for x in g.default.get("--t_masses", None).split(";")]
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

g.message(tscale, tmasses)

g.default.push_verbose("block_cg", True)
a0 = g.qcd.scalar.action.hermitian_kernel.mass_term(
    g.qcd.scalar.action.hermitian_kernel.complement(
        U, 
        g.qcd.gauge.algebra_laplace_polynomial(U, tscale, tmasses), 
        [odd]*4,
        [even]*4
    ),
    g.algorithms.inverter.block_cg({"eps": 1e-15, "maxiter": 300})
)

# conjugate momenta
p_mom = g.group.cartesian(U)

#g.message(f" - {a0.__name__}")
a0.draw(U + p_mom, rng)
a0.assert_gradient_error(rng, U + p_mom, U + p_mom, 1e-3, 1e-8)


# wilson action
a1 = g.qcd.gauge.action.iwasaki(beta)
g.message(f" - {a1.__name__}")

# a1.assert_gradient_error(rng, U_dbl, U_dbl, 1e-3, 1e-8)


def hamiltonian():
    return a0(U + p_mom) + a1(U)

def add(a, b):
    return [g(x+y) for x, y in zip(a, b)]

# molecular dynamics
sympl = g.algorithms.integrator.symplectic

# create OMF2 and OMF2_force_gradient integrators
p_mom2 = g.copy(p_mom)
U2 = g.copy(U)
_iq_imp = sympl.update_q(U, lambda: a0.gradient(U2 + p_mom, p_mom), "Q")
_ip_inner_imp = sympl.update_p(
    p_mom, lambda: add(a1.gradient(U, U), a0.gradient(U + p_mom2, U)), "P_inner"
)
ip_inner_imp = sympl.implicit_update(p_mom, p_mom2, _ip_inner_imp, eps=implicit_eps)
iq_imp = sympl.implicit_update(U, U2, _iq_imp, eps=implicit_eps)

integrator = sympl.OMF2(noutersteps, ip_inner_imp, iq_imp)
integrator0 = sympl.OMF2(1, ip_inner_imp, iq_imp)
# there seems to be a bug in noutersteps > 1 for implicit integrators!

# integrator
g.message(f"Integration scheme:\n{integrator}")

# test error scaling of integrator
h0 = hamiltonian()

U0 = g.copy(U)
p_mom0 = g.copy(p_mom)

integrator0(tau / noutersteps)
h1 = hamiltonian()
integrator0(-tau / noutersteps)
eps2 = sum([g.norm2(u - u2) for u, u2 in zip(U + p_mom, U0 + p_mom0)]) / sum(
    [g.norm2(u) for u in U0 + p_mom0]
)
g.message(f"dH tau={tau} {integrator.__name__} dH={h1-h0:.3g} reversibility={eps2**0.5:.2g}")

#sys.exit(0)

# reset and start HMC
g.copy(U, U0)
g.copy(p_mom, p_mom0)


# metropolis
metro = g.algorithms.markov.metropolis(rng)

# MD units
g.message(f"tau = {tau} MD units")


def hmc():
    a0.draw(U + p_mom, rng)
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

    g.save(f"{root}/ckpoint_lat.{i}", U, g.format.nersc())

