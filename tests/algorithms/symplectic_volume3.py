#!/usr/bin/env python3
#
# Symplecticity test v3: the IMPLICIT case -- a momentum potential that is
# not quadratic in (U, p) jointly and depends on U.
#
# Hamiltonian on T*SU(3) (per link; p = dU U^dag / i, the right-trivialized
# momentum of v1/v2):
#
#   H(U, p) = K(U, p) + V(U) ,   K = 1/2 <A(U)^-1 p, p>  (+ optional (k/4)s^2),
#
# with the U-dependent positive endomorphism of the fiber
#
#   A(U) = I + lambda/2 (Ad_U + Ad_U^T) ,   Ad_U(x) = U x U^dag ,
#
# a symmetric adjoint-action distortion of the fiber metric: su(3)-
# preserving, eigenvalues of 1/2(Ad+Ad^T) in [-1,1]  =>  A > 0 for
# lambda < 1.  K is quadratic in p but U-dependent: the drift velocity
# xi = dK/dp = A(U)^-1 p depends on U, so the implicit midpoint
#
#   p*  :  p* = p - (dt/2) [ dK/dp (U*, p*) + dK/dU (U*, p*) + dV/dU (U*) ]
#   U*  =  U exp(i dt p* / 2) ,      U' = U exp(i dt dK/dp (U*, p*))
#   p'  =  2 p* - p
#
# is a genuine nonlinear (per-link 8x8) Newton solve -- the whole integrator,
# solve included, lives in the black box  integrate(state) -> state1.
#
# Analytic dK/dU (derived from dK = <A^-1 p, dp> - 1/2 <x, (dA) x> with
# x = A^-1 p and d(U^dag x U) = Q^dag x U + U^dag x Q):
#   dK/dU = (lambda/2) [ herm0(-i U x U^dag x) - herm0(-i x U x U^dag) ].
# It is verified against finite differences at startup.
#
# The kinetic can also carry a non-quadratic-in-p correction
#   K = s + (kappa/4) s^2 ,  s = 1/2 <A(U)^-1 p, p> ,
# (U-independent when lambda = 0 -> left-invariant, still genuinely implicit
# because the drift velocity (1 + (kappa/2) s) p* is nonlinear in p*).
#
# KEY FINDING
# -----------
# The geometric-midpoint implicit midpoint is symplectic to O(dt^3) when the
# kinetic is left-invariant (lambda = 0, only V(U) varies or K(p) is
# non-quadratic), but drops to O(dt) when the kinetic itself depends on U
# (lambda > 0): the U-dependent kinetic breaks the generated/left-invariant
# structure.  The lambda > 0 map is still the CORRECT implicit midpoint
# (all midpoint equations hold to machine precision); the O(dt) is a
# property of the method + geometric lifting, not a bug.
#
# Tests (4 links, 1 closed site, 64 dim; v2 harness, FD-only):
#   T0   free flow: implicit midpoint == exact flow              [exact]
#   T1a  quadratic K(p) + V(U): implicit midpoint                 [O(dt^3)]
#   T1b  non-quadratic K(p) = s + (k/4)s^2 + V(U): implicit mid. [O(dt^3)]
#   T2   U-dependent kinetic A(U): correct implicit midpoint
#        (midpoint equations to machine precision), O(dt) near-symplectic
#   T3   leapfrog, A frozen @ old U     (drift is not the flow of
#        1/2<A(U)^-1 p,p>, which also moves p)                   [FAILS]
#   T4   leapfrog, A @ explicit midpoint                         [FAILS]
#   T5   explicit midpoint (lambda>0 and lambda=0)               [FAILS]
#   T6   GPT's own implicit_update (Picard) machinery in the box
#        (tests/algorithms/integrators.py pattern): T6a quadratic
#        K -> the staggered fixed point IS the plain leapfrog
#        [EXACT];  T6b A(U) -> Picard converges but the staggered
#        fixed point is O(dt) non-symplectic  [FAILS, reported]
#
import gpt as g
import numpy as np
from numpy.linalg import svd

rng = np.random.default_rng(20250913)
DT = 0.1


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
# v2 harness (canonical symplectic form + FD pullback), copied for isolation
# ---------------------------------------------------------------------------
class symplectic_harness:
    def __init__(self, G):
        self.G = G
        self.c = 8
        q = np.zeros((8, 8, 8))
        for b in range(8):
            for c2 in range(8):
                comm = 1j * (G[b] @ G[c2] - G[c2] @ G[b])
                for d in range(8):
                    q[b, c2, d] = 2 * np.real(np.trace(G[d] @ comm))
        self.q = q  # i[G_b,G_c] = sum_d q_bc^d G_d

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
                        M[
                            (mu2 * Ns + s2) * 2 * self.c + self.c + r, col
                        ] = 2 * np.real(np.trace(self.G[r] @ dP))
        return M

    def pullback(self, U, p, integrate, eps=1e-6):
        M = self.jacobian_fd(U, p, integrate, eps)
        Om0 = self.omega(U, p)
        U1, p1 = integrate((U, p))
        Om1 = self.omega(U1, p1)
        lhs = M.T @ Om1 @ M
        res = np.abs(lhs - Om0).max() / np.abs(Om0).max()
        return res, np.linalg.det(M)


# ---------------------------------------------------------------------------
# per-link physics at one closed site (pure numpy)
# ---------------------------------------------------------------------------
class physics:
    def __init__(self, G, NL, lam, c, B, U0, p0, kappa=0.0):
        self.G = G
        self.NL = NL
        self.lam = lam
        self.c = c
        self.B = B
        self.U0 = U0
        self.p0 = p0
        self.kappa = kappa  # non-quadratic-in-p kinetic correction

    # V = sum_mu c_mu ReTr(B_mu U_mu);  dV = 2ReTr(F dC)
    def V(self, Um):
        return sum(
            self.c[mu] * np.real(np.trace(self.B[mu] @ Um[mu])) for mu in range(self.NL)
        )

    def F(self, Um):
        r = np.empty_like(Um)
        for mu in range(self.NL):
            r[mu] = herm0(
                1j * self.c[mu] / 4 * (Um[mu] @ self.B[mu] - self.B[mu] @ Um[mu].conj().T)
            )
        return r

    # A(U) = I + lam * 1/2 (Ad_U + Ad_U^T),  Ad_U(x) = U x U^dag
    # (symmetric adjoint-action distortion;  A > 0 for lam < 1)
    def Aact1(self, mu, Us, x):
        # (I + lam S_U)(x),  S_U(x) = 1/2 (U x U^dag + U^dag x U)
        if self.lam == 0.0:
            return x
        Ud = Us.conj().T
        return x + self.lam * 0.5 * (Us @ x @ Ud + Ud @ x @ Us)

    def A8(self, mu, Us):
        # 8x8 matrix of the operator A(U_mu) in the G basis (symmetric)
        M = np.eye(8)
        if self.lam == 0.0:
            return M
        for a in range(8):
            s = self.Aact1(mu, Us, self.G[a]) - self.G[a]
            for r in range(8):
                M[r, a] += 2 * np.real(np.trace(self.G[r] @ s))
        return M

    def Ainv1(self, mu, Us, y):
        M = self.A8(mu, Us)
        yc = np.array([2 * np.real(np.trace(y @ self.G[a])) for a in range(8)])
        coef = np.linalg.solve(M, yc)
        return sum(coef[a] * self.G[a] for a in range(8))

    def K1(self, mu, Us, p):
        # s = 1/2 <A^-1 p, p> = ReTr(p A^-1 p);  K = s + (kappa/4) s^2
        s = np.real(np.trace(p @ self.Ainv1(mu, Us, p)))
        return s + 0.25 * self.kappa * s * s

    def K(self, Um, pm):
        return sum(self.K1(mu, Um[mu], pm[mu]) for mu in range(self.NL))

    def gradp_K(self, Um, pm):
        # dK/dp = (1 + (kappa/2) s) A^-1 p ,  s = ReTr(p A^-1 p)
        r = np.empty_like(pm)
        for mu in range(self.NL):
            s = np.real(np.trace(pm[mu] @ self.Ainv1(mu, Um[mu], pm[mu])))
            r[mu] = (1.0 + 0.5 * self.kappa * s) * self.Ainv1(mu, Um[mu], pm[mu])
        return r

    def gradU_K(self, Um, pm):
        # analytic dK/dU for A = I + lam/2 (Ad + Ad^T):
        #   G = (lam/2) [ herm0(-i U x U^dag x) - herm0(-i x U x U^dag) ],
        #   x = A^-1 p
        r = np.empty_like(pm)
        for mu in range(self.NL):
            if self.lam == 0.0:
                r[mu] = 0.0
                continue
            U = Um[mu]
            Ud = U.conj().T
            x = self.Ainv1(mu, U, pm[mu])
            r[mu] = (self.lam / 2) * (
                herm0(-1j * U @ x @ Ud @ x) - herm0(-1j * x @ U @ x @ Ud)
            )
        return r

    def gradU_K_fd(self, Um, pm, e=1e-6):
        r = np.empty_like(pm)
        for mu in range(self.NL):
            acc = np.zeros((3, 3), dtype=np.complex128)
            for a in range(8):
                Upp = Um.copy()
                Upp[mu] = expm_iH(self.G[a], e) @ Um[mu]
                Umm = Um.copy()
                Umm[mu] = expm_iH(self.G[a], -e) @ Um[mu]
                acc += ((self.K(Upp, pm) - self.K(Umm, pm)) / (2 * e)) * self.G[a]
            r[mu] = acc
        return r

    def check_gradients(self):
        e = 1e-6
        U = self.U0
        p = self.p0
        for mu in range(self.NL):
            a = random_hermitian_su3()
            Up = U.copy()
            Up[mu] = expm_iH(a, e) @ U[mu]
            Um = U.copy()
            Um[mu] = expm_iH(a, -e) @ U[mu]
            fdV = (self.V(Up) - self.V(Um)) / (2 * e)
            anV = 2 * np.real(np.trace(a @ self.F(U)[mu]))
            assert abs(fdV - anV) < 1e-6 * max(1.0, abs(fdV)), (fdV, anV)
            pp = p.copy()
            pp[mu] = p[mu] + e * a
            pm_ = p.copy()
            pm_[mu] = p[mu] - e * a
            fdKp = (self.K(U, pp) - self.K(U, pm_)) / (2 * e)
            anKp = 2 * np.real(np.trace(a @ self.gradp_K(U, p)[mu]))
            assert abs(fdKp - anKp) < 1e-6 * max(1.0, abs(fdKp)), (fdKp, anKp)
            if self.lam > 0:
                fdKu = (self.K(Up, p) - self.K(Um, p)) / (2 * e)
                anKu = 2 * np.real(np.trace(a @ self.gradU_K(U, p)[mu]))
                fdKu_ref = 2 * np.real(np.trace(a @ self.gradU_K_fd(U, p)[mu]))
                assert abs(fdKu - fdKu_ref) < 1e-5, "gradU_K_fd inconsistent"
                assert abs(fdKu - anKu) < 1e-5 * max(1.0, abs(fdKu)), (fdKu, anKu)


# ---------------------------------------------------------------------------
# black boxes: one-step maps on (U, p), (NL, 1, 3, 3)
# ---------------------------------------------------------------------------
class integrator_box:
    def __init__(self, ph, scheme, dt=DT, newton_tol=1e-13, newton_max=60):
        global DT
        DT = dt  # box methods read the module-global DT
        self.ph = ph
        self.scheme = scheme
        self.newton_tol = newton_tol
        self.newton_max = newton_max
        self.stats = {"solves": 0, "iters": 0, "maxit": 0, "maxres": 0.0}
        self.last_midpoint = None

    def _residual(self, U, p, v):
        # implicit-midpoint residual in the drift velocity v (8-dim per link):
        #   U* = U exp(i dt v / 2),  p* = A(U*) v
        #   r  = p* - p + (dt/2)( dK/dU (U*, p*) + F(U*) )
        ph = self.ph
        Ustar = [expm_iH(v[mu], DT / 2) @ U[mu] for mu in range(ph.NL)]
        pstar = np.stack([ph.Aact1(mu, Ustar[mu], v[mu]) for mu in range(ph.NL)])
        r = pstar - p + 0.5 * DT * (ph.gradU_K(np.stack(Ustar), pstar) + ph.F(np.stack(Ustar)))
        return Ustar, pstar, r

    def newton_midpoint(self, U, p):
        # Newton on the 8*L drift velocity v;  returns (v*, U*, p*)
        ph = self.ph
        v = np.empty_like(p)  # initial guess v = p  (exact when A = I)
        for mu in range(ph.NL):
            v[mu] = p[mu]
        r = None
        for it in range(self.newton_max):
            Ustar, pstar, r = self._residual(U, p, v)
            if np.abs(r).max() < self.newton_tol:
                self.stats["solves"] += 1
                self.stats["iters"] += it + 1
                self.stats["maxit"] = max(self.stats["maxit"], it + 1)
                self.stats["maxres"] = max(self.stats["maxres"], float(np.abs(r).max()))
                return v, Ustar, pstar
            J = np.zeros((8 * ph.NL, 8 * ph.NL))
            e = 1e-7
            for mu in range(ph.NL):
                for b in range(8):
                    vp = v.copy()
                    vp[mu] = v[mu] + e * ph.G[b]
                    vm = v.copy()
                    vm[mu] = v[mu] - e * ph.G[b]
                    dr = (self._residual(U, p, vp)[2][mu] - self._residual(U, p, vm)[2][mu]) / (2 * e)
                    for a in range(8):
                        J[mu * 8 + a, mu * 8 + b] = 2 * np.real(np.trace(ph.G[a] @ dr))
            rc = np.zeros(8 * ph.NL)
            for mu in range(ph.NL):
                for a in range(8):
                    rc[mu * 8 + a] = 2 * np.real(np.trace(ph.G[a] @ r[mu]))
            dxc = np.linalg.solve(J, rc)
            v = v.reshape(-1, 3, 3) - np.einsum("ma,aij->mij", dxc.reshape(ph.NL, 8), ph.G)
        raise RuntimeError(f"Newton not converged (res {np.abs(r).max():.2e})")

    def __call__(self, state):
        U, p = state
        U = U[:, 0]  # (NL,3,3) : single closed site
        p = p[:, 0]
        U1, p1 = self._step(U, p)
        return U1[:, None], p1[:, None]

    def newton_midpoint_p(self, U, p):
        # p*-parameterized Newton, for lam = 0 (A = I, gradp_K U-independent):
        #   v* = gradp_K(p*),  U* = U exp(i dt v*/2)
        #   r(p*) = p* - p + (dt/2) F(U*)
        ph = self.ph
        x = p.copy()
        r = None
        for it in range(self.newton_max):
            vstar = ph.gradp_K(x, x)  # U-independent when lam=0
            Ustar = [expm_iH(vstar[mu], DT / 2) @ U[mu] for mu in range(ph.NL)]
            r = x - p + 0.5 * DT * ph.F(np.stack(Ustar))
            if np.abs(r).max() < self.newton_tol:
                self.stats["solves"] += 1
                self.stats["iters"] += it + 1
                self.stats["maxit"] = max(self.stats["maxit"], it + 1)
                self.stats["maxres"] = max(self.stats["maxres"], float(np.abs(r).max()))
                return vstar, Ustar, x
            J = np.zeros((8 * ph.NL, 8 * ph.NL))
            e = 1e-7
            for mu in range(ph.NL):
                for b in range(8):
                    xp = x.copy(); xp[mu] = x[mu] + e * ph.G[b]
                    xm = x.copy(); xm[mu] = x[mu] - e * ph.G[b]
                    dr = (self._res_p(U, p, xp)[2][mu] - self._res_p(U, p, xm)[2][mu]) / (2 * e)
                    for a in range(8):
                        J[mu * 8 + a, mu * 8 + b] = 2 * np.real(np.trace(ph.G[a] @ dr))
            rc = np.zeros(8 * ph.NL)
            for mu in range(ph.NL):
                for a in range(8):
                    rc[mu * 8 + a] = 2 * np.real(np.trace(ph.G[a] @ r[mu]))
            dxc = np.linalg.solve(J, rc)
            x = x.reshape(-1, 3, 3) - np.einsum("ma,aij->mij", dxc.reshape(ph.NL, 8), ph.G)
        raise RuntimeError(f"Newton-p not converged (res {np.abs(r).max():.2e})")

    def _res_p(self, U, p, x):
        vstar = self.ph.gradp_K(x, x)
        Ustar = [expm_iH(vstar[mu], DT / 2) @ U[mu] for mu in range(self.ph.NL)]
        r = x - p + 0.5 * DT * self.ph.F(np.stack(Ustar))
        return vstar, Ustar, r

    def _step(self, U, p):
        ph = self.ph
        L = ph.NL

        if self.scheme == "IMPMID_P":
            assert self.ph.lam == 0.0, "IMPMID_P requires lam=0 (U-independent dK/dp)"
            vstar, Ustar, pstar = self.newton_midpoint_p(U, p)
            U1 = np.stack([expm_iH(vstar[mu], DT) @ U[mu] for mu in range(L)])
            p1 = 2 * pstar - p
            self.last_midpoint = (vstar, Ustar, pstar)
            return U1, p1

        if self.scheme == "IMPMID":
            vstar, Ustar, pstar = self.newton_midpoint(U, p)
            U1 = np.stack([expm_iH(vstar[mu], DT) @ U[mu] for mu in range(L)])
            p1 = 2 * pstar - p
            self.last_midpoint = (vstar, Ustar, pstar)  # diagnostics
            return U1, p1

        if self.scheme == "LEAPFROG_AFROZEN":
            # generated method for H_frozen = 1/2 <A(U_old)^-1 p, p> + V(U)
            F0 = ph.F(U)
            phalf = np.stack([p[mu] - 0.5 * DT * F0[mu] for mu in range(L)])
            U1 = np.stack([expm_iH(ph.gradp_K(U, phalf)[mu], DT) @ U[mu] for mu in range(L)])
            F1 = ph.F(U1)
            p1 = np.stack([phalf[mu] - 0.5 * DT * F1[mu] for mu in range(L)])
            return U1, p1

        if self.scheme == "NAIVE_AU_MIDPOINT":
            # leapfrog, A evaluated at the explicit midpoint: drift velocity
            # 0.5 (A(U)^-1 + A(U_m)^-1) p_half;  at lam=0 = plain leapfrog
            F0 = ph.F(U)
            phalf = np.stack([p[mu] - 0.5 * DT * F0[mu] for mu in range(L)])
            Um = [expm_iH(phalf[mu], DT / 2) @ U[mu] for mu in range(L)]
            v = np.stack(
                [
                    0.5 * (ph.gradp_K(U, phalf)[mu] + ph.gradp_K(Um, phalf)[mu])
                    for mu in range(L)
                ]
            )
            U1 = np.stack([expm_iH(v[mu], DT) @ U[mu] for mu in range(L)])
            F1 = ph.F(U1)
            p1 = np.stack([phalf[mu] - 0.5 * DT * F1[mu] for mu in range(L)])
            return U1, p1

        if self.scheme == "EXPMID":
            # explicit midpoint, A frozen at old U:
            #   U* = U exp(i dt/2 A^-1 p)
            #   p* = (I + dt/2 A^-1)^-1 (p - dt/2 F(U*))     (linear: K frozen)
            #   U1 = U* exp(i dt/2 A^-1 p) ;  p1 = 2 p* - p
            Minv = [np.linalg.inv(ph.A8(mu, U[mu])) for mu in range(L)]
            v0 = np.empty_like(p)  # A^-1 p as 3x3 matrices
            for mu in range(L):
                pc = np.array([2 * np.real(np.trace(p[mu] @ ph.G[a])) for a in range(8)])
                xc = Minv[mu] @ pc
                v0[mu] = sum(xc[a] * ph.G[a] for a in range(8))
            Ustar = [expm_iH(v0[mu], DT / 2) @ U[mu] for mu in range(L)]
            Fstar = ph.F(Ustar)
            pstar = np.empty_like(p)
            for mu in range(L):
                rc = np.array(
                    [
                        2
                        * np.real(np.trace((p[mu] - 0.5 * DT * Fstar[mu]) @ ph.G[a]))
                        for a in range(8)
                    ]
                )
                coef = np.linalg.solve(np.eye(8) + 0.5 * DT * Minv[mu], rc)
                pstar[mu] = sum(coef[a] * ph.G[a] for a in range(8))
            U1 = np.stack([expm_iH(v0[mu], DT / 2) @ Ustar[mu] for mu in range(L)])
            p1 = np.stack([2 * pstar[mu] - p[mu] for mu in range(L)])
            return U1, p1

        raise ValueError(self.scheme)


# ---------------------------------------------------------------------------
# GPT implicit_update machinery (Picard) in the black box
# ---------------------------------------------------------------------------
# Following tests/algorithms/integrators.py:  an explicit update whose force
# reads the GUESS lattice f2 of (possibly other) updates is wrapped in
# sympl.implicit_update, which Picard-iterates it (directions +1/-1 are
# assigned by add_directions inside leap_frog).  The force here reads BOTH
# guess lattices, so the kicks are implicit in (U2, p2) and the drift in U2.
#
# A cold start (U2 := U, p2 := p at step entry) makes the one-step map
# single-step (no stale warm start from a previous call), which the FD
# harness requires.  The resulting fixed point is a "staggered" implicit
# kick-drift-kick (first kick and drift velocity at (U0, p_old)/(U*, p_a)),
# NOT the implicit midpoint -- the harness classifies it.
class gpt_expr_physics:
    # GPT-lattice versions of the v3 physics (Neumann series for A(U)^-1,
    # ratio lam/2 < 1)
    def __init__(self, ph, U, p):
        self.ph = ph
        self.U = U
        self.p = p
        self.ca = U[0].otype.cartesian()
        self.I = [g.identity(U[0]) for _ in range(ph.NL)]
        self.B = [g.lattice(U[0]) for _ in range(ph.NL)]
        for mu in range(ph.NL):
            self.B[mu][:] = ph.B[mu]

    def F(self, Ul):
        r = [
            g(
                g.qcd.gauge.project.traceless_hermitian(
                    1j * self.ph.c[mu] / 4 * (Ul[mu] * self.B[mu] - self.B[mu] * g.adj(Ul[mu]))
                )
            )
            for mu in range(self.ph.NL)
        ]
        for f in r:
            f.otype = self.ca
        return r

    def Ainv(self, Umu, xl):
        # (I + (lam/2) T)^-1 x,  T(x) = U x U^dag + U^dag x U
        # Neumann: sum_k (-lam/2)^k T^k x ;  contraction ratio = lam (T norm <= 2)
        r = g(xl)
        term = g(xl)
        for k in range(1, 35):
            term = g(Umu * term * g.adj(Umu) + g.adj(Umu) * term * Umu)
            r = r + (-(self.ph.lam / 2)) ** k * term
        r = g(r)
        r.otype = self.ca
        return r

    def dKdp(self, Ul, pl):
        # drift velocity A(U)^-1 p  (kappa = 0)
        return [self.Ainv(Ul[mu], pl[mu]) for mu in range(self.ph.NL)]

    def dKdU(self, Ul, pl):
        # (lam/2) herm0(i (W2 - W1)),  W1 = U x U^dag x, W2 = x U x U^dag,
        # x = A(U)^-1 p   (same as the numpy gradU_K, FD-verified)
        r = []
        for mu in range(self.ph.NL):
            x = self.Ainv(Ul[mu], pl[mu])
            W1 = Ul[mu] * x * g.adj(Ul[mu]) * x
            W2 = x * Ul[mu] * x * g.adj(Ul[mu])
            t = g(0.5 * self.ph.lam * 1j * (W2 - W1))
            t = g(0.5 * (t + g.adj(t)))
            t = g(t - self.I[mu] * g.trace(t) / 3)
            t.otype = self.ca  # otype must be set on the EVALUATED lattice
            r.append(t)
        return r


def make_gpt_implicit_box(ph, grid=None, picard_eps=1e-13):
    assert ph.kappa == 0.0, "GPT-implicit box implemented for kappa = 0"
    grid = grid or g.grid([1, 1, 1, 1], g.double)
    x = (0, 0, 0, 0)
    NL = ph.NL
    U = [g.mcolor(grid) for _ in range(NL)]
    p = [g.group.cartesian(U[0]) for _ in range(NL)]
    U2 = [g.lattice(U[0]) for _ in range(NL)]   # guess lattices
    p2 = [g.lattice(p[0]) for _ in range(NL)]
    gpe = gpt_expr_physics(ph, U, p)
    sympl = g.algorithms.integrator.symplectic

    _ip = sympl.update_p(p, lambda: [gpe.dKdU(U2, p2)[mu] + gpe.F(U2)[mu] for mu in range(NL)], "ip")
    _iq = sympl.update_q(U, lambda: gpe.dKdp(U2, p), "iq")
    ip_imp = sympl.implicit_update(p, p2, _ip, eps=picard_eps, tag="P")
    iq_imp = sympl.implicit_update(U, U2, _iq, eps=picard_eps, tag="Q")
    step = sympl.leap_frog(1, ip_imp, iq_imp)

    stats = {"picard_res": 0.0}

    def integrate(state):
        U0, p0 = state
        # pre-step snapshots (for the Picard-residual check)
        Upre = [g.lattice(U[0]) for _ in range(NL)]
        ppre = [g.lattice(p[0]) for _ in range(NL)]
        for mu in range(NL):
            U[mu][x] = U0[mu, 0]
            p[mu][x] = p0[mu, 0]
            Upre[mu][x] = U0[mu, 0]
            ppre[mu][x] = p0[mu, 0]
        # cold start: guesses := current state (single-step, no stale warm start)
        for mu in range(NL):
            g.copy(U2[mu], U[mu])
            g.copy(p2[mu], p[mu])
        step(DT)
        # Picard fixed-point residual of the LAST op (ip -1):
        #   p2 == p_a - (dt/2)(dKdU + F)(U2, p2),  p_a = ppre - (dt/2)(dKdU+F)(Upre, ppre)
        Ga = [gpe.dKdU(U2, p2)[mu] + gpe.F(U2)[mu] for mu in range(NL)]
        G0 = [gpe.dKdU(Upre, ppre)[mu] + gpe.F(Upre)[mu] for mu in range(NL)]
        pa = [g(ppre[mu] - 0.5 * DT * G0[mu]) for mu in range(NL)]
        res = 0.0
        for mu in range(NL):
            r = g(p2[mu] - (pa[mu] - 0.5 * DT * Ga[mu]))
            res = max(res, float(g.norm2(r)) ** 0.5)
        stats["picard_res"] = max(stats["picard_res"], res)
        return (
            np.stack([U[mu][x].array for mu in range(NL)], axis=0)[:, None],
            np.stack([p[mu][x].array for mu in range(NL)], axis=0)[:, None],
        )

    return integrate, stats, gpe


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def pullback_res(h, ph, scheme, dt, U, p):
    box = integrator_box(ph, scheme, dt)
    M = h.jacobian_fd(U, p, box)
    Om0 = h.omega(U, p)
    U1, p1 = box((U, p))
    Om1 = h.omega(U1, p1)
    res = np.abs(M.T @ Om1 @ M - Om0).max() / np.abs(Om0).max()
    return res, np.linalg.det(M), box


def test_exact(h, label, ph, scheme, dt, U, p, tol=1e-8):
    res, detj, box = pullback_res(h, ph, scheme, dt, U, p)
    newton = ""
    if box.stats["solves"]:
        st = box.stats
        newton = f"   newton {st['solves']}x avg {st['iters'] / st['solves']:.1f} it, maxres {st['maxres']:.0e}"
    print(f"{label:<34} pullback {res:.3e}  det J {detj:.10f}  [exact]{newton}")
    assert res < tol, (label, res)
    assert abs(detj - 1) < 1e-6, (label, detj)


def test_highorder(h, label, ph, scheme, U, p, dt=0.1):
    # implicit midpoint on a non-abelian group: symplectic to O(dt^3) --
    # residual must be small AND shrink ~8x per dt halving
    r1, d1, _ = pullback_res(h, ph, scheme, dt, U, p)
    r2, d2, box = pullback_res(h, ph, scheme, dt / 2, U, p)
    ratio = r1 / r2 if r2 > 0 else float("inf")
    st = box.stats
    newton = f"   newton {st['solves']}x avg {st['iters'] / st['solves']:.1f} it, maxres {st['maxres']:.0e}" if st["solves"] else ""
    print(f"{label:<34} pullback {r1:.3e} (dt={dt})  {r2:.3e} (dt/2)  x{ratio:.1f}  det {d1:.9f}{newton}")
    assert r1 < 5e-3, (label, r1, "residual too large")
    assert 4.0 <= ratio <= 16.0, (label, ratio, "not O(dt^2..dt^4) scaling")


def test_fail(h, label, ph, scheme, dt, U, p, floor=1e-2):
    res, detj, _ = pullback_res(h, ph, scheme, dt, U, p)
    print(f"{label:<34} pullback {res:.3e}  det J {detj:.10f}  [must FAIL]")
    assert res > floor, (label, res, "expected to fail clearly")


def im_equation_residuals(ph, U, p, box, dt):
    # verify the box's IMPMID map satisfies the 5 implicit-midpoint equations:
    #   (1) p* = A(U*) v            (2) U* = U exp(i dt v / 2)
    #   (3) p* = p - (dt/2) dH/dU (U*,p*)
    #   (4) U1 = U exp(i dt v)      (5) p1 = p - dt dH/dU (U*,p*)
    U1, p1 = box((U[:, None], p[:, None]))  # runs the step, stores the midpoint
    v, Ustar, pstar = box.last_midpoint
    UstarA = np.stack(Ustar)
    dHdu = ph.gradU_K(UstarA, pstar) + ph.F(UstarA)
    L = ph.NL
    res = []
    for mu in range(L):
        res.append(np.abs(ph.Aact1(mu, Ustar[mu], v[mu]) - pstar[mu]).max())   # (1)
        res.append(np.abs(Ustar[mu] - expm_iH(v[mu], dt / 2) @ U[mu]).max())   # (2)
        res.append(np.abs(pstar[mu] - (p[mu] - 0.5 * dt * dHdu[mu])).max())    # (3)
        res.append(np.abs(U1[mu, 0] - expm_iH(v[mu], dt) @ U[mu]).max())       # (4)
        res.append(np.abs(p1[mu, 0] - (p[mu] - dt * dHdu[mu])).max())          # (5)
    return max(res)


def main():
    global DT
    grid = g.grid([1, 1, 1, 1], g.double)
    gen = g.mcolor(grid).otype.cartesian().generators(grid.precision.complex_dtype)
    G = np.stack([np.asarray(gen[a].array) for a in range(8)])
    h = symplectic_harness(G)

    NL = 4
    c = [0.5, 0.8, 0.3, 0.6]
    B = [random_hermitian_su3() for _ in range(NL)]
    U0 = np.stack([random_su3() for _ in range(NL)])
    p0 = np.stack([random_hermitian_su3() for _ in range(NL)])
    U = U0[:, None].copy()
    p = p0[:, None].copy()
    dt = 0.1

    ph_quad = physics(G, NL, 0.0, c, B, U0, p0)
    ph_au = physics(G, NL, 0.5, c, B, U0, p0)
    ph_au.check_gradients()
    emin = min(np.linalg.eigvalsh(ph_au.A8(mu, U0[mu])).min() for mu in range(NL))
    print(f"A(U) fiber eigenvalue min = {emin:.4f}  (lambda = {ph_au.lam})")
    assert emin > 0.4

    print("")
    print("--- T0 calibration: implicit midpoint on the free flow is the exact flow ---")
    ph_free = physics(G, NL, 0.0, [0.0] * NL, B, U0, p0)
    test_exact(h, "T0 free flow (IM == exact)", ph_free, "IMPMID", dt, U, p)

    print("")
    print("--- T1 positive: left-invariant kinetic + V(U): IM symplectic to O(dt^3) ---")
    test_highorder(h, "T1a IMPMID quadratic + V(U)", ph_quad, "IMPMID", U, p, dt)
    # non-quadratic-in-p kinetic  K = s + (k/4) s^2  (still U-independent /
    # left-invariant), so the implicit midpoint stays O(dt^3); the drift
    # velocity (1+(k/2)s) p* makes the step genuinely implicit
    ph_nq = physics(G, NL, 0.0, c, B, U0, p0, kappa=0.8)
    ph_nq.check_gradients()
    test_highorder(h, "T1b IMPMID non-quad K(p) + V(U)", ph_nq, "IMPMID_P", U, p, dt)

    print("")
    print("--- T2 U-dependent kinetic A(U): Newton + correct implicit midpoint ---")
    # the geometric-midpoint IM with a U-dependent kinetic is only O(dt)
    # symplectic on SU(3) (it loses the generated/left-invariant structure);
    # what we certify here is that the implicit solve produces the correct
    # implicit-midpoint map (all 5 midpoint equations to machine precision)
    box = integrator_box(ph_au, "IMPMID", dt)
    im_res = im_equation_residuals(ph_au, U0, p0, box, dt)
    st = box.stats
    print(f"T2 A(U) IM: newton {st['solves']}x avg {st['iters']/st['solves']:.1f} it, "
          f"maxres {st['maxres']:.0e};  IM-equation residual {im_res:.2e}")
    assert st["maxres"] < 1e-12, "Newton did not converge"
    assert im_res < 1e-10, (im_res, "map is not the implicit midpoint")
    r1, _, _ = pullback_res(h, ph_au, "IMPMID", dt, U, p)
    r2, _, _ = pullback_res(h, ph_au, "IMPMID", dt / 2, U, p)
    print(f"T2 A(U) IM: pullback {r1:.3e} (dt)  {r2:.3e} (dt/2)  x{r1/r2:.1f}  "
          f"[O(dt) near-symplectic, reported]")
    assert 1.5 <= r1 / r2 <= 3.0, (r1 / r2, "expected O(dt) scaling")

    print("")
    print("--- T6 GPT implicit_update (Picard) machinery in the black box ---")
    # the GPT box cross-checks its GPT-expression physics against the numpy one
    box6a, st6a, gpe6a = make_gpt_implicit_box(ph_quad, picard_eps=1e-12)
    box6b, st6b, gpe6b = make_gpt_implicit_box(ph_au, picard_eps=1e-12)
    def gpe_check(gpe, phh, tag):
        Ux = [g.mcolor(g.grid([1,1,1,1], g.double)) for _ in range(phh.NL)]
        px = [g.group.cartesian(Ux[0]) for _ in range(phh.NL)]
        for mu in range(phh.NL):
            Ux[mu][0,0,0,0] = phh.U0[mu]
            px[mu][0,0,0,0] = phh.p0[mu]
        dF = max(np.abs(gpe.F(Ux)[mu][0,0,0,0].array - phh.F(phh.U0)[mu]).max() for mu in range(phh.NL))
        dKdp = max(np.abs(gpe.dKdp(Ux,px)[mu][0,0,0,0].array - phh.gradp_K(phh.U0, phh.p0)[mu]).max() for mu in range(phh.NL))
        dKdU = max(np.abs(gpe.dKdU(Ux,px)[mu][0,0,0,0].array - phh.gradU_K(phh.U0, phh.p0)[mu]).max() for mu in range(phh.NL))
        print(f"T6 {tag} GPT-vs-numpy physics: dF {dF:.1e}  d(dK/dp) {dKdp:.1e}  d(dK/dU) {dKdU:.1e}")
        assert max(dF, dKdp, dKdU) < 1e-9
    gpe_check(gpe6a, ph_quad, "(quad)")
    gpe_check(gpe6b, ph_au, "(A(U))")

    # T6a: quadratic K (lambda=0): the staggered Picard fixed point coincides
    # with the plain leapfrog (kick@U0, drift by half-kicked p, kick@U1)
    # -> must be exactly symplectic
    DT = dt
    M = h.jacobian_fd(U, p, box6a)
    Om0 = h.omega(U, p)
    U1x, p1x = box6a((U, p))
    Om1 = h.omega(U1x, p1x)
    res6a = np.abs(M.T @ Om1 @ M - Om0).max() / np.abs(Om0).max()
    print(f"T6a GPT-Picard quadratic (== leapfrog): pullback {res6a:.3e}  det {np.linalg.det(M):.10f}  "
          f"picard res {st6a['picard_res']:.1e}")
    assert st6a["picard_res"] < 1e-11, "Picard did not converge"
    assert res6a < 1e-8, res6a

    # T6b: A(U) kinetic (lambda>0): Picard converges, but the staggered fixed
    # point is O(dt) non-symplectic (first kick at (U0,p0), drift velocity at
    # (U*, p_a) rather than the full midpoint (U*, p*))
    for d in [dt, dt / 2]:
        DT = d
        M = h.jacobian_fd(U, p, box6b)
        Om0 = h.omega(U, p)
        U1x, p1x = box6b((U, p))
        Om1 = h.omega(U1x, p1x)
        r = np.abs(M.T @ Om1 @ M - Om0).max() / np.abs(Om0).max()
        print(f"T6b GPT-Picard A(U), dt={d}: pullback {r:.3e}  det {np.linalg.det(M):.10f}  "
              f"picard res {st6b['picard_res']:.1e}")
        if d == dt:
            r6b1, det6b = r, np.linalg.det(M)
        else:
            r6b2 = r
    assert st6b["picard_res"] < 1e-11, "Picard did not converge"
    assert r6b1 > 1e-3, (r6b1, "GPT-Picard A(U) should be clearly non-symplectic")
    assert 1.5 <= r6b1 / r6b2 <= 3.0, (r6b1 / r6b2, "expected O(dt) scaling")

    print("")
    print("--- negatives: naive schemes must FAIL the pullback ---")
    test_fail(h, "T3 leapfrog, A frozen @ old U", ph_au, "LEAPFROG_AFROZEN", dt, U, p)
    test_fail(h, "T4 leapfrog, A @ explicit midpoint", ph_au, "NAIVE_AU_MIDPOINT", dt, U, p)
    test_fail(h, "T5 explicit midpoint (lam>0)", ph_au, "EXPMID", dt, U, p)
    test_fail(h, "T5 explicit midpoint (lam=0)", ph_quad, "EXPMID", dt, U, p)

    print("")
    print("v3 implicit-case symplecticity tests passed")


if __name__ == "__main__":
    main()
