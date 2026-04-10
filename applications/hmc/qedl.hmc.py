#!/usr/bin/env python3
#
# Authors: Christoph Lehner
#
import gpt as g
import numpy as np
import sys, os

# parameters
tau = g.default.get_float("--tau", 1.0)
root = g.default.get("--root", None)
seed = g.default.get("--seed", "hmc-pure-gauge")
n = g.default.get_int("--n", 1000)

# grid
grid = g.grid([4, 4, 4, 8], g.double)
rng = g.random(seed)

# load state / initialize state
A = [g.real(grid) for _ in range(4)]

fn_try = None
i0 = 0
for i in range(0, n):
    fn = f"{root}/ckpoint_lat.A.{i}"
    if os.path.exists(fn):
        fn_try = fn
        i0 = i + 1

if fn_try is not None:
    rng = g.random(fn_try)
    A0 = g.load(fn_try)
    for mu in range(4):
        A[mu] @= A0[mu]

A_mom = g.group.cartesian(A)  # conjugate momenta

a_qed = g.qcd.gauge.action.non_compact.qed_l(grid)

# no fourier acceleration, only masking (remove spatial zero modes)
diagonal_mass = [[g.real(grid) for _ in range(4)] for _ in range(4)]
for mu in range(4):
    for nu in range(4):
        diagonal_mass[mu][nu][:] = 0
        if mu == nu:
            diagonal_mass[mu][nu][:] = 1
a_mom = g.qcd.scalar.action.fourier_mass_term(diagonal_mass, mask = a_qed.base.mask)

a_mom.draw(A_mom, rng)
a_qed.draw(A, rng)

a_mom.assert_gradient_error(rng, A_mom, A_mom, 1e-4, 1e-8)
a_qed.assert_gradient_error(rng, A, A, 1e-4, 1e-8)

sympl = g.algorithms.integrator.symplectic

def hamiltonian():
    a_qed_v = a_qed(A)
    a_mom_v = a_mom(A_mom)
    return a_qed_v + a_mom_v

ip = sympl.update_p(A_mom, lambda: a_qed.gradient(A, A))
iq = sympl.update_q(A, lambda: a_mom.gradient(A_mom, A_mom))

# integrator
mdint = sympl.OMF4(5, ip, iq)

# test error scaling of integrator
h0 = hamiltonian()

# metropolis
metro = g.algorithms.markov.metropolis(rng)

# MD units
tau = 2.0
g.message(f"tau = {tau} MD units")


def hmc():
    a_mom.draw(A_mom, rng)
    accrej = metro(A)
    h0 = hamiltonian()
    mdint(tau)
    h1 = hamiltonian()
    # print(g(g.fft() * A[0])[0,0,0,1])
    return [accrej(h1, h0), h1 - h0]

p = 2*np.pi * np.array([1,0,0,0]) / np.array(grid.gdimensions)
prop = g(g.exp_ixp(p) * a_qed.propagator()[0][0])
cor_ref = g.slice(prop, 3)

# production
f = open("c1.hmc.dat", "wt")
for i in range(i0, n):
    _, dH = hmc()
    g.message(f"Trajectory {i}, dH={dH}")

    for j in range(4):
        prop = g(g.exp_ixp(p) * g.correlate(A[j], A[j]))
        cor = g.slice(prop, 3)

        f.write(f"{4*i+j} {cor[1].real}\n")
        f.flush()

        if i > 20 and i % 5 == 0:
            o = g.corr_io.writer(f"{root}/{4*i+j}.hmc.dat")
            o.write("ref", cor_ref)
            o.write("cor", cor)
            o.close()
    
    # g.save(f"{root}/ckpoint_lat.A.{i}", A)

    # if g.rank() == 0:
    #     flog = open(f"{root}/ckpoint_lat.A.{i}.log", "wt")
    #     flog.write(f"dH {dH}\n")
    #     flog.close()

    g.barrier()

f.close()
