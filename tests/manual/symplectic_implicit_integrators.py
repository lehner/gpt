#!/usr/bin/env python3
#
# Symplecticity test GPT implicit integrators via finite-difference tests of the pullback and jacobian
#
# The test runs for
# - a factorizable U(1) Hamiltonian
# - a non-factorizable SU(3) Hamiltonian
# - a non-factorizable U(1) Hamiltonian
#
# The test runs on a single node for now and only uses a single lattice site
import gpt as g
import numpy as np
from numpy.linalg import svd

sympl = g.algorithms.integrator.symplectic

rng = np.random.default_rng(20250914)


def random_su3():
    a = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    w, _, vh = svd(a)
    u = w @ vh
    d = np.linalg.det(u)
    return u @ np.diag([1.0, 1.0, 1.0 / d])


def random_hermitian_su3():
    x = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    h = 0.5 * (x + x.conj().T)
    return h - np.trace(h) / 3 * np.eye(3)


def expm_iH(H, x=1.0):
    w, v = np.linalg.eigh(H)
    return v @ np.diag(np.exp(1j * x * w)) @ v.conj().T


def herm0(X):
    H = 0.5 * (X + X.conj().T)
    return H - np.trace(H) / 3 * np.eye(3)


# ---------------------------------------------------------------------------
# v2/v3 harness (canonical symplectic form + FD Jacobian + pullback)
# ---------------------------------------------------------------------------
class symplectic_harness:
    def __init__(self, G):
        # G: (c, d, d) generator array;  d = 3 (su3) or 1 (u1)
        self.G = G
        self.c = G.shape[0]
        q = np.zeros((self.c, self.c, self.c))
        for b in range(self.c):
            for c2 in range(self.c):
                comm = 1j * (G[b] @ G[c2] - G[c2] @ G[b])
                for d in range(self.c):
                    q[b, c2, d] = 2 * np.real(np.trace(G[d] @ comm))
        self.q = q  # i[G_b,G_c] = sum_d q_bc^d G_d  (zero for abelian u1)

    def omega(self, U, p):
        NL, Ns = U.shape[0], U.shape[1]
        n = 2 * NL * Ns * self.c

        def idx(mu, s, a, kind):
            return (mu * Ns + s) * 2 * self.c + a + (self.c if kind == "p" else 0)

        Om = np.zeros((n, n))
        for mu in range(NL):
            for s in range(Ns):
                pc = [2 * np.real(np.trace(p[mu, s] @ self.G[d])) for d in range(self.c)]
                for b in range(self.c):
                    for c2 in range(self.c):
                        Om[idx(mu, s, b, "U"), idx(mu, s, c2, "U")] = np.dot(self.q[b, c2, :], pc)
                        Om[idx(mu, s, b, "U"), idx(mu, s, c2, "p")] = -1.0 if b == c2 else 0.0
                        Om[idx(mu, s, b, "p"), idx(mu, s, c2, "U")] = 1.0 if b == c2 else 0.0
        return Om

    def jacobian_fd(self, U, p, integrate, eps=1e-6):
        NL, Ns = U.shape[0], U.shape[1]
        n = 2 * NL * Ns * self.c
        U1b, p1b = integrate((U, p))
        M = np.zeros((n, n))
        for col in range(n):
            block = col // (2 * self.c)
            mu = block // Ns
            s = block % Ns
            is_U = (col % (2 * self.c)) < self.c
            a = col % self.c
            for sgn in [+1, -1]:
                Uv = U.copy()
                pv = p.copy()
                if is_U:
                    Uv[mu, s] = expm_iH(self.G[a], sgn * eps) @ U[mu, s]
                else:
                    pv[mu, s] = p[mu, s] + sgn * eps * self.G[a]
                U1, p1 = integrate((Uv, pv))
                if sgn > 0:
                    U1p, p1p = U1, p1
                else:
                    U1m, p1m = U1, p1
            for mu2 in range(NL):
                for s2 in range(Ns):
                    dU = (U1p[mu2, s2] - U1m[mu2, s2]) / (2 * eps)
                    dP = (p1p[mu2, s2] - p1m[mu2, s2]) / (2 * eps)
                    C = dU @ np.conj(U1b[mu2, s2].T) / 1j  # right-trivialized
                    for r in range(self.c):
                        M[(mu2 * Ns + s2) * 2 * self.c + r, col] = 2 * np.real(
                            np.trace(self.G[r] @ C)
                        )
                        M[(mu2 * Ns + s2) * 2 * self.c + self.c + r, col] = 2 * np.real(
                            np.trace(self.G[r] @ dP)
                        )
        return M

    def pullback(self, U, p, integrate, eps=1e-6):
        M = self.jacobian_fd(U, p, integrate, eps)
        Om0 = self.omega(U, p)
        U1, p1 = integrate((U, p))
        Om1 = self.omega(U1, p1)
        lhs = M.T @ Om1 @ M
        res = np.abs(lhs - Om0).max() / np.abs(Om0).max()
        return res, np.linalg.det(M)


def random_u1():
    return np.array([[np.exp(1j * rng.normal())]], dtype=np.complex128)


def random_real_u1():
    return np.array([[rng.normal()]], dtype=np.complex128)


def make_implicit_box_aU(integrator, integrator_is_fg, dt, NL, mode, lam=0.5, c=None, picard_eps=1e-13):
    # A(U) kinetic  K = 1/2 <A(U)^-1 p, p>  + V(U),  GPT implicit integrator
    # (fixed add_directions), cold started.  modes:
    #   "su3"    : A(U) = I + lam/2 (Ad_U + Ad_U^T)   (U-dependent, v3 form)
    #   "u1"     : same Ad form -- but Ad is trivial in U(1), so A = 1+lam
    #              is CONSTANT: the box reduces to the plain (rescaled)
    #              integrator and must PASS (a trivial pass, flagged below)
    #   "u1_gen" : genuinely U(1)-dependent metric  a(U) = 1 + lam Re(U)
    #              (B = 1).  Empirically SYMPLECTIC (1 DoF): the drift pair
    #              theta1 = theta + 0.5dt a(theta) pa + 0.5dt a(theta1) pa
    #              alone has dtheta1/dtheta != 1, but in 1 DoF the two
    #              kicks compensate it exactly (det = 1 is one condition).
    #              The compensation does NOT happen in SU(3) ("su3" mode).
    c = c or [0.5, 0.8, 0.3, 0.6]
    grid = g.grid([1, 1, 1, 1], g.double)
    x = (0, 0, 0, 0)
    if mode == "su3":
        U = [g.mcolor(grid) for _ in range(NL)]
    else:
        U = [g.u1(grid) for _ in range(NL)]
    p = [g.group.cartesian(U[0]) for _ in range(NL)]
    U2 = [g.lattice(U[0]) for _ in range(NL)]
    p2 = [g.lattice(p[0]) for _ in range(NL)]
    ca = U[0].otype.cartesian()

    def Im(L):
        return g(g(0.5 * (L - g.adj(L)) / 1j))

    def Re(L):
        return g(g(0.5 * (L + g.adj(L))))

    def Ainv(Umu, xl):
        # (I + (lam/2) T)^-1 x,  T(x) = U x U^dag + U^dag x U  (Neumann)
        r = g(xl)
        term = g(xl)
        for k in range(1, 35):
            term = g(Umu * term * g.adj(Umu) + g.adj(Umu) * term * Umu)
            r = r + (-(lam / 2)) ** k * term
        r = g(r)
        r.otype = ca
        return r

    def velocity():
        if mode == "su3":
            r = [Ainv(U2[mu], p2[mu]) for mu in range(NL)]
        elif mode == "u1":
            r = [g(p2[mu] / (1 + lam)) for mu in range(NL)]
        else:  # u1_gen
            r = [g((g.identity(p2[mu]) + lam * Re(U2[mu])) * p2[mu]) for mu in range(NL)]
        for f in r:
            f.otype = ca
        return r

    def force():
        r = []
        for mu in range(NL):
            if mode == "su3":
                xx = Ainv(U2[mu], p2[mu])
                W1 = U2[mu] * xx * g.adj(U2[mu]) * xx
                W2 = xx * U2[mu] * xx * g.adj(U2[mu])
                dKdU = g(0.5 * lam * 1j * (W2 - W1))
                dKdU = g(0.5 * (dKdU + g.adj(dKdU)))
                dKdU = g(dKdU - g.identity(U2[mu]) * g.trace(dKdU) / 3)
            elif mode == "u1":
                dKdU = g.lattice(p2[mu])
                dKdU[:] = 0
            else:  # u1_gen:  1/2 a'(theta) p^2,  a' = -lam Im(U)
                dKdU = g(0.5 * (-lam) * Im(U2[mu]) * p2[mu] * p2[mu])
            F = g(-c[mu] * Im(U2[mu]))  # V = c Re(B U), B=1:  dV = -c Im(U) dtheta
            r.append(g(dKdU + F))
        for f in r:
            f.otype = ca
        return r

    _ip = sympl.update_p(p, force, "ip")
    _iq = sympl.update_q(U, velocity, "iq")
    ip_imp = sympl.implicit_update(p, p2, _ip, eps=picard_eps, tag="P")
    iq_imp = sympl.implicit_update(U, U2, _iq, eps=picard_eps, tag="Q")

    # for test of implicit integrators
    if integrator_is_fg:
        ip_fg_imp = sympl.implicit_update(
            U + p,
            U2 + p2,
            sympl.update_p_force_gradient(U + U2, _iq, p + p2, _ip, _ip, "P_FG"),
            eps=1e-16,
            tag="FG_P",
        )
        integrator1 = integrator(1, ip_imp, iq_imp, ip_fg_imp)
    else:
        integrator1 = integrator(1, ip_imp, iq_imp)

    def read(lat):
        v = lat[x]
        if hasattr(v, "array"):
            return np.asarray(v.array)
        return np.array([[complex(v)]], dtype=np.complex128)

    def write(lat, a):
        if mode == "su3":
            lat[x] = a
        else:  # u1 singlet: scalar in, 1x1 out
            lat[x] = complex(a[0, 0])

    def integrate(state):
        U0, p0 = state
        for mu in range(NL):
            write(U[mu], U0[mu, 0])
            write(p[mu], p0[mu, 0])
            g.copy(U2[mu], U[mu])  # cold start
            g.copy(p2[mu], p[mu])
        integrator1(dt)
        return (
            np.stack([read(U[mu]) for mu in range(NL)], axis=0)[:, None],
            np.stack([read(p[mu]) for mu in range(NL)], axis=0)[:, None],
        )

    return integrate


g.default.set_verbose("implicit_update", False)  # silence the Picard messages
grid = g.grid([1, 1, 1, 1], g.double)
gen = g.mcolor(grid).otype.cartesian().generators(grid.precision.complex_dtype)
G = np.stack([np.asarray(gen[a].array) for a in range(8)])
h = symplectic_harness(G)

integrators = [
    (sympl.leap_frog, False),
    (sympl.OMF2, False),
    (sympl.OMF4, False),
    (sympl.OMF2_force_gradient, True)
]

NL = 4
B = [random_hermitian_su3() for _ in range(NL)]
U = np.empty((NL, 1, 3, 3), dtype=np.complex128)
p = np.empty((NL, 1, 3, 3), dtype=np.complex128)
for mu in range(NL):
    U[mu, 0] = random_su3()
    p[mu, 0] = random_hermitian_su3()

for integrator, integrator_is_fg in integrators:
    g.message(f"Run test for {integrator.__name__}")
    # ------------------------------------------------------------------
    # A(U) kinetic  K = 1/2 <A(U)^-1 p, p>,  group as a parameter
    # ------------------------------------------------------------------
    # 2 links are enough for the A(U) group mechanics (per-link maps);
    # keeps the expensive su3 Neumann box calls half-size for the suite
    NL_a = 2
    lam = 0.5
    modes = [
        # mode     : A(U)                          expectation
        ("u1", "Ad form -> 1+lam CONSTANT", "pass"),
        ("su3", "Ad form, U-dependent (v3)", "fail"),
        # 1 DoF: det = 1 is a single condition, and the two kicks exactly
        # compensate the staggered double drift -> symplectic even with a
        # genuinely U-dependent metric (unlike SU(3) above)
        ("u1_gen", "a(theta) = 1+lam cos(theta)", "pass"),
    ]
    for mode, desc, expct in modes:
        if mode == "su3":
            Ga = np.stack([np.asarray(gen[a].array) for a in range(8)])
            Ua = np.empty((NL_a, 1, 3, 3), dtype=np.complex128)
            pa = np.empty((NL_a, 1, 3, 3), dtype=np.complex128)
            for mu in range(NL_a):
                Ua[mu, 0] = random_su3()
                pa[mu, 0] = random_hermitian_su3()
        else:
            # orthonormal under the harness 2ReTr metric: 2Re(G G) = 1
            Ga = np.array([[[1.0 / np.sqrt(2.0)]]], dtype=np.complex128)
            Ua = np.empty((NL_a, 1, 1, 1), dtype=np.complex128)
            pa = np.empty((NL_a, 1, 1, 1), dtype=np.complex128)
            for mu in range(NL_a):
                Ua[mu, 0] = random_u1()
                pa[mu, 0] = random_real_u1()
        ha = symplectic_harness(Ga)
        g.message(f"A(U) mode={mode:<7} ({desc})   [expect {expct}]  ({NL_a} links)")
        prev = None
        for dt in [0.1, 0.05, 0.025]:
            box = make_implicit_box_aU(integrator, integrator_is_fg, dt, NL_a, mode, lam=lam)
            res, detj = ha.pullback(Ua, pa, box)
            # the scaling ratio is only meaningful for the failing modes
            g.message(f"  dt={dt:<6} pullback={res:<16.3e} det(J)-1={detj-1:16.3e}")
            if expct == "pass":
                assert res < 1e-6, (mode, dt, res)
                assert abs(detj - 1) < 1e-6, (mode, dt, detj)
            else:
                assert res > 1e-4, (mode, dt, res, "expected clearly non-symplectic")

        g.message()
    g.message()
