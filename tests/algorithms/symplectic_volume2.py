#!/usr/bin/env python3
#
# Symplecticity test v2: full symplectic-form pullback  F*omega = omega
# through a black-box one-step integrator.
#
# Black-box contract
# ------------------
#   integrate(state) -> state1,   state = (U, p), numpy (NL, Ns, 3, 3)
# The ENTIRE integrator (scheme, force evaluation, implicit solves, ...)
# lives inside `integrate`.  The map must be a one-step map on T*SU(3)^{NL*Ns}
# whose continuous limit is the Hamilton flow of some H(U, p) with respect to
# the CANONICAL symplectic form of T*SU(3), with p the left-trivialized
# canonical momentum (drift direction xi = dH/dp in the left frame).  H may
# depend on U in the kinetic term, e.g.  H = <A(U)^-1 p, p>/2 + V(U);  the
# pullback statement below is independent of the form of H.  (Note: for
# U-dependent A(U) a naive kick-drift scheme is no longer symplectic -- the
# kick's fiber Jacobian is I - dt (d_U A^-1) p -- and this test detects that.)
#
# The test
# -------
# In the right-trivialized frame {i G_a U} x {G_a} (per link and site) --
# right because GPT's drift is the LEFT action U' = exp(i dt p) U, so
# p = dU U^dag / i -- with structure constants  q_bc^d = 2ReTr(G_d i[G_b,G_c])
# and p_a = <p, G_a>, the canonical form is
#   omega((xi1, eta1), (xi2, eta2))
#       = <eta1, xi2> - <eta2, xi1> + sum_a p_a (q xi1 xi2)^a ,
# i.e. per 16-dim block  Omega = [[+(q.p), -I], [I, 0]]  (xi first).
#   (The +1 MC coefficient was fixed empirically by requiring the exact free
#    flow -- a definite symplectic map -- to satisfy the pullback identity.)
# A map Phi with frame Jacobian M (input frame -> output frame) is symplectic
# iff
#   M^T Omega(Phi x) M = Omega(x) .
# This is strictly stronger than volume preservation (det M = 1): e.g. the
# fiber rotation (U, p) -> (U, Ad_R p) has det = 1 but is NOT symplectic
# (T4b below); forward Euler fails the pullback too (T4).
#
# Validity per test: F*omega = omega certifies a map ON A CLOSED SYMPLECTIC
# SUBSYSTEM.  A single site is closed only for site-local H; with the gauge
# (Iwasaki) action the kick force at site x reads links at x+/-mu, so a
# single-site map with frozen environment is a non-symplectic section
# (T2 documents this: the residual is O(dt) even for a correct integrator).
# The gauge pullback test is therefore run on the full small lattice (T3).
#
# Finite differences only (no AD): the black box is opaque by design.
#
# T0   pure-numpy exact free flow, LARGE dt: calibrates frame/omega/algebra
# T1   1 site, 4 links, site-local H: leap_frog / OMF2 / OMF4 / sympl-Euler
# T4   1 site, 4 links, site-local H: forward Euler (pullback FAILS)
# T4b  synthetic fiber rotation: det=1 (volume) but pullback FAILS
# T2   2x2x1x1, Iwasaki, single site, frozen random environment (negative)
# T3   2x2x1x1, Iwasaki, full lattice 256 dim: leap_frog / OMF2 / OMF4 (positive)
#
import gpt as g
import numpy as np
from numpy.linalg import svd

rng = np.random.default_rng(20250912)


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
    # exp(i x H) for Hermitian H
    w, v = np.linalg.eigh(H)
    return v @ np.diag(np.exp(1j * x * w)) @ v.conj().T


# ---------------------------------------------------------------------------
# canonical symplectic form and pullback harness (pure numpy)
# ---------------------------------------------------------------------------
class symplectic_harness:
    def __init__(self, G):
        # G: (8,3,3) Hermitian generators, 2ReTr(G_a G_b) = delta_ab
        self.G = G
        self.c = 8
        q = np.zeros((8, 8, 8))
        for b in range(8):
            for c2 in range(8):
                comm = 1j * (G[b] @ G[c2] - G[c2] @ G[b])
                for d in range(8):
                    q[b, c2, d] = 2 * np.real(np.trace(G[d] @ comm))
        self.q = q  # q_bc^d, real: i[G_b,G_c] = sum_d q_bc^d G_d

    def omega(self, U, p):
        NL, Ns = U.shape[0], U.shape[1]
        n = 2 * NL * Ns * self.c

        def idx(mu, s, a, kind):
            return (mu * Ns + s) * 2 * self.c + a + (self.c if kind == "p" else 0)

        Om = np.zeros((n, n))
        for mu in range(NL):
            for s in range(Ns):
                pc = [2 * np.real(np.trace(p[mu, s] @ self.G[d])) for d in range(8)]
                for b in range(8):
                    for c2 in range(8):
                        # MC block coefficient is +1 (fixed empirically by
                        # requiring the exact free flow to be symplectic)
                        Om[idx(mu, s, b, "U"), idx(mu, s, c2, "U")] = np.dot(
                            self.q[b, c2, :], pc
                        )
                        Om[idx(mu, s, b, "U"), idx(mu, s, c2, "p")] = (
                            -1.0 if b == c2 else 0.0
                        )
                        Om[idx(mu, s, b, "p"), idx(mu, s, c2, "U")] = (
                            1.0 if b == c2 else 0.0
                        )
        return Om

    def jacobian_fd(self, U, p, integrate, eps=1e-6):
        NL, Ns = U.shape[0], U.shape[1]
        n = 2 * NL * Ns * self.c
        U1b, p1b = integrate((U, p))
        M = np.zeros((n, n))
        for col in range(n):
            # interleaved (mu, s, [C, p]) layout, consistent with omega()/idx()
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
                    # RIGHT-trivialize the image tangent (GPT drift is a left
                    # action U' = exp(i dt p) U, so p = dU U^dag / i):
                    C = dU @ np.conj(U1b[mu2, s2].T) / 1j
                    for r in range(self.c):
                        M[(mu2 * Ns + s2) * 2 * self.c + r, col] = 2 * np.real(
                            np.trace(self.G[r] @ C)
                        )
                        M[
                            (mu2 * Ns + s2) * 2 * self.c + self.c + r, col
                        ] = 2 * np.real(np.trace(self.G[r] @ dP))
        return M

    def pullback(self, U, p, integrate, eps=1e-6):
        # M maps input-frame vectors (at x) to output-frame vectors (at Phi x):
        # w = M v.  (Phi*omega)(v1,v2) = omega(Phi x, M v1, M v2)
        #        = v1^T (M^T Omega(Phi x) M) v2,  so symplecticity is
        #            M^T Omega(Phi x) M = Omega(x).
        M = self.jacobian_fd(U, p, integrate, eps)
        Om0 = self.omega(U, p)
        U1, p1 = integrate((U, p))
        Om1 = self.omega(U1, p1)
        lhs = M.T @ Om1 @ M
        res = np.abs(lhs - Om0).max() / np.abs(Om0).max()
        return res, np.linalg.det(M)


# ---------------------------------------------------------------------------
# GPT black boxes
# ---------------------------------------------------------------------------
def make_site_local_box(NL, scheme, dt, c, B):
    # 1 site, NL links, V = sum_mu c_mu ReTr(B_mu U_mu), analytic force
    grid = g.grid([1, 1, 1, 1], g.double)
    x = (0, 0, 0, 0)
    U = [g.mcolor(grid) for _ in range(NL)]
    p = [g.group.cartesian(U[0]) for _ in range(NL)]
    Bl = [g.lattice(U[0]) for _ in range(NL)]
    for mu in range(NL):
        Bl[mu][:] = B[mu]
    ca = U[0].otype.cartesian()

    def force():
        r = [
            g(
                g.qcd.gauge.project.traceless_hermitian(
                    1j * c[mu] / 4 * (U[mu] * Bl[mu] - Bl[mu] * g.adj(U[mu]))
                )
            )
            for mu in range(NL)
        ]
        for f in r:
            f.otype = ca  # must be the algebra otype, not the group's
        return r

    sympl = g.algorithms.integrator.symplectic
    ip = sympl.update_p(p, force, "P")
    iq = sympl.update_q(U, lambda: p, "Q")
    if scheme == "leap_frog":
        step = sympl.leap_frog(1, ip, iq)
    elif scheme == "OMF2":
        step = sympl.OMF2(1, ip, iq)
    elif scheme == "OMF4":
        step = sympl.OMF4(1, ip, iq)
    elif scheme == "SEULER":
        # symplectic Euler (kick with old U, drift with NEW p): symplectic
        from gpt.algorithms.integrator import euler

        step = sympl.symplectic_base("SEULER")
        step.add((p, euler(p, force, -1), True, "P"), 1, 0)
        step.add((U, euler(U, lambda: p, +1), True, "Q"), 1, 0)
        step.add_directions()
    elif scheme == "FORWARD_EULER":
        # naive forward Euler: drift AND kick both from the OLD state --
        # NOT symplectic (and not volume preserving)
        def step(tau):
            F = force()  # force at the current (old) U
            for mu in range(NL):
                U[mu] @= g.matrix.exp(1j * tau * p[mu]) * U[mu]
            for mu in range(NL):
                p[mu] @= p[mu] - tau * F[mu]

    def integrate(state):
        U0, p0 = state
        for mu in range(NL):
            U[mu][x] = U0[mu, 0]
            p[mu][x] = p0[mu, 0]
        step(dt)
        return (
            np.stack([U[mu][x].array for mu in range(NL)], axis=0)[:, None],
            np.stack([p[mu][x].array for mu in range(NL)], axis=0)[:, None],
        )

    return integrate


def make_iwasaki_box(grid_size, scheme, dt, probe_site=None, env_U=None, env_p=None):
    # Iwasaki gauge field; state = full lattice, or just probe_site with the
    # environment frozen at (env_U, env_p)
    grid = g.grid(list(grid_size), g.double)
    pos = [tuple(t) for t in np.ndindex(*grid.gdimensions)]
    NL = 4
    U = [g.mcolor(grid) for _ in range(NL)]
    p = [g.group.cartesian(U[0]) for _ in range(NL)]
    act = g.qcd.gauge.action.iwasaki(6.0)
    sympl = g.algorithms.integrator.symplectic
    ip = sympl.update_p(p, lambda: act.gradient(U, U), "P")
    iq = sympl.update_q(U, lambda: p, "Q")
    if scheme == "leap_frog":
        step = sympl.leap_frog(1, ip, iq)
    elif scheme == "OMF2":
        step = sympl.OMF2(1, ip, iq)
    elif scheme == "OMF4":
        step = sympl.OMF4(1, ip, iq)

    env_U = env_U if env_U is not None else U
    env_p = env_p if env_p is not None else p

    def integrate(state):
        U0, p0 = state
        # environment (all sites, or all sites except the probe) at reference
        if probe_site is None:
            for mu in range(NL):
                for k, x in enumerate(pos):
                    U[mu][x] = U0[mu, k]
                    p[mu][x] = p0[mu, k]
        else:
            kstar = pos.index(probe_site)
            for mu in range(NL):
                for k, x in enumerate(pos):
                    if k == kstar:
                        U[mu][x] = U0[mu, 0]
                        p[mu][x] = p0[mu, 0]
                    else:
                        U[mu][x] = env_U[mu][x].array
                        p[mu][x] = env_p[mu][x].array
        step(dt)
        if probe_site is None:
            out_U = np.empty((NL, len(pos), 3, 3), dtype=np.complex128)
            out_p = np.empty((NL, len(pos), 3, 3), dtype=np.complex128)
            for mu in range(NL):
                for k, x in enumerate(pos):
                    out_U[mu, k] = U[mu][x].array
                    out_p[mu, k] = p[mu][x].array
            return out_U, out_p
        kstar = pos.index(probe_site)
        out_U = np.empty((NL, 1, 3, 3), dtype=np.complex128)
        out_p = np.empty((NL, 1, 3, 3), dtype=np.complex128)
        for mu in range(NL):
            out_U[mu, 0] = U[mu][probe_site].array
            out_p[mu, 0] = p[mu][probe_site].array
        return out_U, out_p

    return integrate


# ---------------------------------------------------------------------------
# T0: machinery calibration -- exact free flow at large dt (pure numpy)
# ---------------------------------------------------------------------------
def t0(h):
    g.message("T0: exact free flow, large dt (machinery calibration)")
    NL, Ns = 4, 1
    U = np.empty((NL, Ns, 3, 3), dtype=np.complex128)
    p = np.empty((NL, Ns, 3, 3), dtype=np.complex128)
    for mu in range(NL):
        for s in range(Ns):
            U[mu, s] = random_su3()
            p[mu, s] = random_hermitian_su3()
    dt = 0.9  # large: eigenvalues of i dt p approach the branch cut

    def integrate(state):
        U0, p0 = state
        U1 = np.empty_like(U0)
        for mu in range(U0.shape[0]):
            for s in range(U0.shape[1]):
                U1[mu, s] = expm_iH(p0[mu, s], dt) @ U0[mu, s]
        return U1, p0.copy()

    res, detj = h.pullback(U, p, integrate)
    g.message(f"T0: pullback residual {res:.3e}   det J {detj:.10f}")
    assert res < 1e-8, res
    assert abs(detj - 1) < 1e-8, detj
    # omega sanity: antisymmetric, non-degenerate
    Om = h.omega(U, p)
    assert np.abs(Om + Om.T).max() < 1e-14
    sign, logdet = np.linalg.slogdet(Om)
    assert sign > 0, "omega degenerate"


# ---------------------------------------------------------------------------
# T1: 1 site, 4 links, site-local H  (closed subsystem: positive test)
# ---------------------------------------------------------------------------
def t1(h):
    g.message("T1: 1 site, 4 links, site-local H (closed subsystem)")
    NL = 4
    c = [0.5, 0.8, 0.3, 0.6]
    B = [random_hermitian_su3() for _ in range(NL)]
    U = np.empty((NL, 1, 3, 3), dtype=np.complex128)
    p = np.empty((NL, 1, 3, 3), dtype=np.complex128)
    for mu in range(NL):
        U[mu, 0] = random_su3()
        p[mu, 0] = random_hermitian_su3()
    dt = 0.1
    for scheme in ["leap_frog", "OMF2", "OMF4", "SEULER"]:
        integrate = make_site_local_box(NL, scheme, dt, c, B)
        res, detj = h.pullback(U, p, integrate)
        g.message(f"T1 {scheme:<10}: pullback {res:.3e}   det J {detj:.10f}")
        assert res < 1e-6, res
        assert abs(detj - 1) < 1e-6, detj


# ---------------------------------------------------------------------------
# T4: naive forward Euler (drift+kick from OLD state) on the closed subsystem:
#     NOT symplectic -> pullback must FAIL.
# T4b: synthetic fiber rotation (U, p) -> (U, Ad_R p): volume preserving
#     (det = 1) but NOT symplectic -> shows the pullback test is strictly
#     stronger than a volume test.
# ---------------------------------------------------------------------------
def t4(h, ref_res):
    g.message("T4: forward Euler, site-local H (pullback must FAIL)")
    NL = 4
    c = [0.5, 0.8, 0.3, 0.6]
    B = [random_hermitian_su3() for _ in range(NL)]
    U = np.empty((NL, 1, 3, 3), dtype=np.complex128)
    p = np.empty((NL, 1, 3, 3), dtype=np.complex128)
    for mu in range(NL):
        U[mu, 0] = random_su3()
        p[mu, 0] = random_hermitian_su3()
    dt = 0.1
    integrate = make_site_local_box(NL, "FORWARD_EULER", dt, c, B)
    res, detj = h.pullback(U, p, integrate)
    g.message(f"T4 FWD_EUL  : pullback {res:.3e}   det J {detj:.10f}")
    assert res > 100 * ref_res, "forward Euler pullback should fail clearly"


def t4b(h, ref_res):
    g.message("T4b: fiber rotation (U,Ad_R p) -- det=1 but pullback must FAIL")
    NL, Ns = 4, 1
    U = np.empty((NL, Ns, 3, 3), dtype=np.complex128)
    p = np.empty((NL, Ns, 3, 3), dtype=np.complex128)
    for mu in range(NL):
        U[mu, 0] = random_su3()
        p[mu, 0] = random_hermitian_su3()
    # a genuine SU(3) rotation R -> Ad_R is an SO(8) rotation on the fiber
    R = expm_iH(random_hermitian_su3(), 0.7)

    def integrate(state):
        U0, p0 = state
        p1 = np.empty_like(p0)
        for mu in range(NL):
            for s2 in range(Ns):
                p1[mu, s2] = R @ p0[mu, s2] @ R.conj().T
        return U0.copy(), p1

    res, detj = h.pullback(U, p, integrate)
    g.message(f"T4b ROTATE  : pullback {res:.3e}   det J {detj:.10f}")
    assert abs(detj - 1) < 1e-6, "fiber rotation must preserve volume"
    assert res > 100 * ref_res, "fiber rotation pullback should fail clearly"


# ---------------------------------------------------------------------------
# T2: Iwasaki, single site, frozen random environment (NOT closed: negative)
# ---------------------------------------------------------------------------
def t2(h, ref_res, grid_size, dt=0.05):
    g.message(f"T2: Iwasaki, single site, frozen environment ({grid_size})")
    # random environment reference on the full lattice
    env_U = np.empty((4, np.prod(grid_size), 3, 3), dtype=np.complex128)
    env_p = np.empty((4, np.prod(grid_size), 3, 3), dtype=np.complex128)
    for mu in range(4):
        for k in range(np.prod(grid_size)):
            env_U[mu, k] = random_su3()
            env_p[mu, k] = random_hermitian_su3()
    # probe site in a full-lattice box with everything frozen at env reference
    grid = g.grid(list(grid_size), g.double)
    pos = [tuple(t) for t in np.ndindex(*grid.gdimensions)]
    xstar = pos[0]
    envU_lats = [g.mcolor(grid) for _ in range(4)]
    envp_lats = [g.group.cartesian(envU_lats[0]) for _ in range(4)]
    for mu in range(4):
        for k, x in enumerate(pos):
            envU_lats[mu][x] = env_U[mu, k]
            envp_lats[mu][x] = env_p[mu, k]

    integrate = make_iwasaki_box(grid_size, "leap_frog", dt, probe_site=xstar,
                                 env_U=envU_lats, env_p=envp_lats)
    U = np.empty((4, 1, 3, 3), dtype=np.complex128)
    p = np.empty((4, 1, 3, 3), dtype=np.complex128)
    for mu in range(4):
        U[mu, 0] = random_su3()
        p[mu, 0] = random_hermitian_su3()
    res, detj = h.pullback(U, p, integrate)
    g.message(f"T2 frozen site: pullback {res:.3e}   det J {detj:.10f} (both != 0/1 expected)")
    # the restricted map is neither volume- nor symplectic-preserving
    assert res > 100 * ref_res, "frozen-site map should fail the pullback test"


# ---------------------------------------------------------------------------
# T3: Iwasaki, full small lattice (closed: positive test)
# ---------------------------------------------------------------------------
def t3(h, grid_size, dt=0.05):
    g.message(f"T3: Iwasaki, full lattice {grid_size} (closed subsystem)")
    Ns = int(np.prod(grid_size))
    U = np.empty((4, Ns, 3, 3), dtype=np.complex128)
    p = np.empty((4, Ns, 3, 3), dtype=np.complex128)
    for mu in range(4):
        for k in range(Ns):
            U[mu, k] = random_su3()
            p[mu, k] = random_hermitian_su3()
    for scheme in ["leap_frog", "OMF2", "OMF4"]:
        integrate = make_iwasaki_box(grid_size, scheme, dt)
        res, detj = h.pullback(U, p, integrate)
        g.message(f"T3 {scheme:<10}: pullback {res:.3e}   det J {detj:.10f}")
        assert res < 1e-4, res
        assert abs(detj - 1) < 1e-5, detj


def main():
    # generator frame from GPT (Tr(G_a G_b) = 1/2 delta_ab, Hermitian)
    grid = g.grid([1, 1, 1, 1], g.double)
    gen = g.mcolor(grid).otype.cartesian().generators(grid.precision.complex_dtype)
    G = np.stack([np.asarray(gen[a].array) for a in range(8)])
    h = symplectic_harness(G)

    t0(h)
    t1(h)
    ref_res = None
    # reference residual scale from T1 (leap_frog) for the negative tests
    NL = 4
    c = [0.5, 0.8, 0.3, 0.6]
    B = [random_hermitian_su3() for _ in range(NL)]
    U = np.empty((NL, 1, 3, 3), dtype=np.complex128)
    p = np.empty((NL, 1, 3, 3), dtype=np.complex128)
    for mu in range(NL):
        U[mu, 0] = random_su3()
        p[mu, 0] = random_hermitian_su3()
    ref_res, _ = h.pullback(U, p, make_site_local_box(NL, "leap_frog", 0.1, c, B))
    t4(h, ref_res)
    t4b(h, ref_res)
    t2(h, ref_res, (2, 2, 1, 1))
    t3(h, (2, 2, 1, 1))
    g.message("Symplectic pullback tests v2 passed")


if __name__ == "__main__":
    main()
