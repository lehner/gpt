#!/usr/bin/env python3
#
# Authors: Christoph Lehner
#
# Symmetric HMC with left and right updates
#
import gpt as g
import sys, os
import numpy as np

noutersteps = 1
beta = 2.95
seed = "test2"

rng = g.random(seed)

U = g.load("hmc_2p95/ckpoint_lat.110")
grid = U[0].grid

sm = []
for rho in [0.12]:
    for mu in range(4):
        for p in [g.even, g.odd]:
            sm.append(g.qcd.gauge.smear.local_stout(rho=rho, dimension=mu, checkerboard=p))

for s in sm:
    U = s.inv(U)

g.message("Integration variable plaquette:", g.qcd.gauge.plaquette(U))


a0 = g.qcd.scalar.action.mass_term()
a1 = g.qcd.gauge.action.iwasaki(beta)
mom = g.group.cartesian(U)

a_log_det = None
for s in sm: # sm = [ Sx Sy Sz St ] ->   f(U) -> f(Sx(U)) -> ... -> f(Sx(Sy(Sz(St(U)))))
    a1 = a1.transformed(s)
    if a_log_det is None:
        a_log_det = s.action_log_det_jacobian()
    else:
        a_log_det = a_log_det.transformed(s) + s.action_log_det_jacobian()

a1 = a1 + a_log_det


def hamiltonian():
    a_gauge = a1(U)
    return a0(mom) + a_gauge

sympl = g.algorithms.integrator.symplectic

ip = sympl.update_p(mom, lambda: a1.gradient(U, U))
iq = sympl.update_q(U, lambda: a0.gradient(mom, mom))

mdint = sympl.OMF4(noutersteps, ip, iq)

V = g.mcolor(grid)
rng.element(V)

force0 = a1.gradient(U, U)

Ut = g.qcd.gauge.transformed(U, V)

force1 = a1.gradient(Ut, Ut)

for mu in range(4):
    err2 = g.norm2(g(V*force0[mu]*g.adj(V)) - force1[mu])
    g.message(mu, err2)
    assert err2 < 1e-23, "Gauge transformation of force does not hold"

def gauge_fixing():
    opt = g.algorithms.optimize
    cg = opt.non_linear_cg(
        maxiter=1000,
        eps=1e-8,
        step=0.1,
        line_search=opt.line_search_quadratic,
        beta=opt.polak_ribiere,
        max_abs_step=0.1
    )
    f = g.qcd.gauge.fix.landau(U)
    fa = opt.fourier_accelerate.inverse_phat_square(U[0].grid, f)
    assert cg(fa)(V, V)

gauge_fixing()
V0 = g.copy(V)

def analyze_single(mom0, mom, tag):
    res = []
    for mu in range(len(mom)):
        A = g.inner_product(mom0[mu], mom[mu])
        B = g.inner_product(mom0[mu], mom0[mu])
        C = g.inner_product(mom[mu], mom[mu])
        res.append( (A / B**0.5 / C**0.5).real )
        
    g.message(tag, np.mean(res), np.std(res) / len(mom)**0.5)

def analyze(mom0, mom, tag):
    analyze_single(mom0, mom, f"{tag} full")
    analyze_single(
        [g.sum(x) for x in mom0],
        [g.sum(x) for x in mom],
        f"{tag} p0"
    )

    for i in range(1, 4):
        # diagonal
        p = [2*np.pi / grid.gdimensions[nu] * i for nu in range(4)]
        analyze_single(
            [g.sum(g.exp_ixp(p=np.array(p)) * x) for x in mom0],
            [g.sum(g.exp_ixp(p=np.array(p)) * x) for x in mom],
            f"{tag} pd{i}"
        )

        # momentum lorentz index breaks symmetry, so should
        # separate parallel and orthogonal

        p = [[2*np.pi / grid.gdimensions[0] * i if mu == nu else 0 for nu in range(4)] for mu in range(4)]
        
        # parallel
        left = [
            g.sum(g.exp_ixp(p=np.array(p[j])) * mom0[j]) for j in range(4)
        ]

        right = [
            g.sum(g.exp_ixp(p=np.array(p[j])) * mom[j]) for j in range(4)
        ]
        
        analyze_single(left, right, f"{tag} pp{i}")

        # orthogonal
        left = [
            g.sum(g.exp_ixp(p=np.array(p[k])) * mom0[j]) for j in range(4) for k in range(j) if k != j
        ]

        right = [
            g.sum(g.exp_ixp(p=np.array(p[k])) * mom[j]) for j in range(4) for k in range(j) if k != j
        ]
        
        analyze_single(left, right, f"{tag} po{i}")
    
    
def hmc(t):
    U0 = g.copy(U)

    rng.normal_element(mom)

    mom0_gf = [g(V0*mom[mu]*g.adj(V0)) for mu in range(4)]
    mom0 = g.copy(mom)
    
    h0 = hamiltonian()
    nsteps = int(t / 0.2) + 1
    g.message("nsteps",nsteps)
    for _ in range(nsteps):
        mdint(t / nsteps)
    h1 = hamiltonian()
    g.message("dH",h1-h0)

    gauge_fixing()
    mom_gf = [g(V*mom[mu]*g.adj(V)) for mu in range(4)]

    analyze(mom0, mom, f"{t} orig")
    analyze(mom0_gf, mom_gf, f"{t} landau")

    g.copy(U, U0)
    

for tau in list(np.arange(0.1, 4.0, 0.1)):
    hmc(tau)



