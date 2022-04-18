#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2022
#
import gpt as g
import numpy as np

# load configuration
rng = g.random("test")
grid = g.grid([8, 8, 8, 16], g.double)
V = rng.element(g.mcolor(grid))


# fake propagator
U = g.mspincolor(grid)
D = g.mspincolor(grid)
S = g.mspincolor(grid)
rng.cnormal([U, D, S])


def uud_two_point(Q1, Q2, kernel):
    #
    # eps(a,b,c) eps(a',b',c') (kernel)_alpha'beta' (kernel)_alphabeta Pp_gammagamma' Q2_beta'beta_b'b
    # (Q1_alpha'alpha_a'a Q1_gamma'gamma_c'c - Q1_alpha'gamma_a'c Q1_gamma'alpha_c'a)
    #
    # =
    #
    # eps(a,b,c) eps(a',b',c') (kernel)_alpha'beta' (kernel)_alphabeta Pp_gammagamma' Q2_beta'beta_b'b
    # Q1_alpha'alpha_a'a Q1_gamma'gamma_c'c
    # -
    # eps(a,b,c) eps(a',b',c') (kernel)_alpha'beta' (kernel)_alphabeta Pp_gammagamma' Q2_beta'beta_b'b
    # Q1_alpha'gamma_a'c Q1_gamma'alpha_c'a
    #
    # =
    #
    # eps(c,a,b) eps(c',a',b') (Q1 kernel)_alpha'beta_a'a (kernel Q2)^Tspin_betaalpha'_b'b  Tr_S[Pp Q1_c'c]
    # +
    # eps(a,b,c) eps(a',b',c')  Pp_gammagamma'
    #  (Q1 kernel)_gamma'beta_c'c (kernel Q2)^Tspin_betaalpha'_b'b Q1_alpha'gamma_a'a
    #
    # =
    #
    # Tr_S[diquark(Q1 kernel, kernel Q2)_{cc'}] Tr_S[Pp Q1_c'c]
    # +
    # diquark(Q1 kernel, kernel Q2)_{gamma'alpha', aa'} Pp_gammagamma'
    #   Q1_alpha'gamma_a'a
    #
    # =
    #
    # Tr_C[Tr_S[diquark(Q1 kernel, kernel Q2)] Tr_S[Q1 Pp]]
    # +
    # Tr[ Q1 Pp diquark(Q1 kernel, kernel Q2) ]
    #
    #
    # with
    #
    # diquark(Q1,Q2)_{gammagamma', cc'} = eps(c,a,b) eps(c',a',b') (Q1)_gammabeta_a'a (Q2)^Tspin_betagamma'_b'b
    #
    dq = g.qcd.baryon.diquark(g(Q1 * kernel), g(kernel * Q2))
    return g(g.color_trace(g.spin_trace(dq) * Q1 + dq * Q1))


def proton(Q1, Q2):
    C = 1j * g.gamma[1].tensor() * g.gamma[3].tensor()
    Gamma = C * g.gamma[5].tensor()
    Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.5
    return g.trace(uud_two_point(Q1, Q2, Gamma) * Pp)


# test gauge invariance
proton1 = proton(U, D)
proton2 = proton(g(V * U * g.adj(V)), g(V * D * g.adj(V)))

eps = (g.norm2(proton1 - proton2) / g.norm2(proton1)) ** 0.5
g.message(f"Gauge invariance of proton two-point: {eps}")
assert eps < 1e-13


# omega
def sss_two_point(Q1, kernel):
    dq = g.qcd.baryon.diquark(g(Q1 * kernel), g(kernel * Q1))
    return g(-2.0 * g.color_trace(g.spin_trace(dq) * Q1 + 2.0 * dq * Q1))


def omega(Q1, mu):
    # Note: mu=1 has different sign from mu=0,2 in this way!
    C = 1j * g.gamma[1].tensor() * g.gamma[3].tensor()
    Gamma = C * g.gamma[mu].tensor()
    Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.5
    return g(g.trace(sss_two_point(Q1, Gamma) * Pp))


for mu in range(3):
    omega1 = omega(S, mu)
    omega2 = omega(g(V * S * g.adj(V)), mu)

    eps = (g.norm2(omega1 - omega2) / g.norm2(omega1)) ** 0.5
    g.message(f"Gauge invariance of omega_{mu} two-point: {eps}")
    assert eps < 1e-13
