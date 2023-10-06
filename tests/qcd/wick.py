#!/usr/bin/env python3
import gpt as g

grid = g.grid([8, 8, 8, 16], g.double)
rng = g.random("d")
prop = g.mspincolor(grid)
rng.cnormal(prop)

w = g.qcd.wick()

x, y = w.coordinate(2)

ud_propagators = {
    (x, y): prop[0, 0, 0, 0],
    (y, x): g(g.gamma[5] * g.adj(prop[0, 0, 0, 0]) * g.gamma[5]),
}

u = w.fermion(ud_propagators)
d = w.fermion(ud_propagators)
s = w.fermion(ud_propagators)

na = w.color_index()
nalpha, nbeta = w.spin_index(2)

C = 1j * g.gamma[1].tensor() * g.gamma[3].tensor()
Cg5 = w.spin_matrix(C * g.gamma[5].tensor())
Pp = w.spin_matrix((g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.5)


#####
# Baryon tests
def nucleon_operator(w, u, d, x, alpha, matrix):
    a, b, c = w.color_index(3)
    beta, gamma = w.spin_index(2)
    return w.sum(
        u(x, alpha, a),
        w.sum(
            u(x, beta, b),
            w.epsilon(a, b, c),
            w.sum(matrix(beta, gamma), d(x, gamma, c), gamma),
            beta,
            b,
            c,
        ),
        a,
    )


O = nucleon_operator(w, u, d, x, nalpha, Cg5)

Obar = nucleon_operator(w, u.bar(), d.bar(), y, nbeta, Cg5)

proton_2pt = w.sum(Obar, Pp(nbeta, nalpha), O, nalpha, nbeta)


def uud_two_point(Q1, Q2, kernel):
    dq = g.qcd.baryon.diquark(g(Q1 * kernel), g(kernel * Q2))
    return g(g.color_trace(g.spin_trace(dq) * Q1 + dq * Q1))


def proton(Q1, Q2):
    C = 1j * g.gamma[1].tensor() * g.gamma[3].tensor()
    Gamma = C * g.gamma[5].tensor()
    Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.5
    return g(g.trace(uud_two_point(Q1, Q2, Gamma) * Pp))[0, 0, 0, 0]


C_proton_2pt = proton(prop, prop)
W_proton_2pt = w(proton_2pt, verbose=True)

eps = abs(C_proton_2pt - W_proton_2pt) / abs(C_proton_2pt)
g.message(f"Proton 2pt test: {eps}")
assert eps < 1e-14

# Omega
Cgi = [w.spin_matrix(C * g.gamma[i].tensor()) for i in range(3)]


def sss_two_point(Q1, kernel):
    dq = g.qcd.baryon.diquark(g(Q1 * kernel), g(kernel * Q1))
    return g(-2.0 * g.color_trace(g.spin_trace(dq) * Q1 + 2.0 * dq * Q1))


def omega(Q1, mu):
    C = 1j * g.gamma[1].tensor() * g.gamma[3].tensor()
    Gamma = C * g.gamma[mu].tensor()
    Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.5
    return g(g.trace(sss_two_point(Q1, Gamma) * Pp))[0, 0, 0, 0]


for mu in range(3):
    W_omega_2pt = w(
        w.sum(
            nucleon_operator(w, s, s, x, nalpha, Cgi[mu]),
            Pp(nbeta, nalpha),
            nucleon_operator(w, s.bar(), s.bar(), y, nbeta, Cgi[mu]),
            nalpha,
            nbeta,
        ),
        verbose=True,
    )

    C_omega_2pt = omega(prop, mu)

    eps = abs(C_omega_2pt - W_omega_2pt) / abs(C_omega_2pt)
    g.message(f"Omega_{mu} 2pt test: {eps}")
    assert eps < 1e-14


#####
# Meson tests
def meson(w, ubar, d, x, matrix):
    a = w.color_index()
    alpha, beta = w.spin_index(2)
    return w.sum(ubar(x, alpha, a) * matrix(alpha, beta) * d(x, beta, a), alpha, beta, a)


meson_2pt = meson(w, u.bar(), d, x, Cg5) * meson(w, d.bar(), u, y, Cg5)

W_meson_2pt_all = w(meson_2pt, verbose=True, separate_diagrams=True)
W_meson_2pt = w(meson_2pt)

eps = abs(W_meson_2pt_all[0] - W_meson_2pt) / abs(W_meson_2pt)
g.message(f"Separate diagrams test: {eps}")
assert eps < 1e-14

C_meson_2pt = g(
    -g.trace(
        prop
        * (C * g.gamma[5].tensor() * g.gamma[5].tensor())
        * g.adj(prop)
        * (g.gamma[5].tensor() * C * g.gamma[5].tensor())
    )
)[0, 0, 0, 0]

eps = abs(C_meson_2pt - W_meson_2pt) / abs(W_meson_2pt)
g.message(f"Meson 2pt test: {eps}")
assert eps < 1e-14
