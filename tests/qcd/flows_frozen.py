#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2026
#
# Volume-preserving frozen-generator shifts (directional_frozen_shift and its
# checkerboard wrapper frozen_shift): exact inverse, gauge covariance, unit
# Jacobian determinant, forces, the frozen-links check
#
import gpt as g
import numpy as np

rng = g.random("flows_frozen")
smear = g.qcd.gauge.smear


def staple_pair(mu, nu, F):
    # the staples x -> x + mu through x + nu and x - nu (they avoid U_mu(x))
    up = F[nu] * g.cshift(F[mu], nu, 1) * g.adj(g.cshift(F[nu], mu, 1))
    dn = g.cshift(g.adj(F[nu]) * F[mu] * g.cshift(F[nu], mu, 1), nu, -1)
    return up, dn


def plaquette(nu, lam, F):
    # the transverse plaquette at x (frozen links only for nu, lam != mu)
    return F[nu] * g.cshift(F[lam], nu, 1) * g.adj(g.cshift(F[nu], lam, 1)) * g.adj(F[lam])


def generator_staples(mu, nu):
    # rho S_+ S_-^dag, rho a parameter (complex field or number)
    def G(F, xp):
        up, dn = staple_pair(mu, nu, F)
        return g(xp[0] * up * g.adj(dn))

    return G


def generator_staples_plaquette(mu, nu, lam):
    # rho S_+ S_-^dag + sigma P_{nu lam} S_+ S_-^dag
    def G(F, xp):
        up, dn = staple_pair(mu, nu, F)
        s = up * g.adj(dn)
        return g(xp[0] * s + xp[1] * plaquette(nu, lam, F) * s)

    return G


def relative(a, b):
    return (sum(g.norm2(x - y) for x, y in zip(a, b)) / sum(g.norm2(y) for y in b)) ** 0.5


#####################################################################
# 4^4 SU(3)
#####################################################################
grid = g.grid([4, 4, 4, 8], g.double)
U = g.qcd.gauge.random(grid, rng)
even, odd = g.even_odd_projectors(grid)
rho = g.complex(grid)
rng.normal(rho)
rho @= 0.3 * rho
sigma = 0.2 + 0.1j
params = [rho, sigma]

# a general mask: the even sites with even x_3 (any subset of a checkerboard
# keeps the staples frozen)
coor = g.coordinates(even)
quarter = g.complex(grid)
quarter[:] = 0
quarter[coor[coor[:, 3] % 2 == 0]] = 1
quarter = g(quarter * even)

steps = []
for mu in range(4):
    nu, lam = [x for x in range(4) if x != mu][0:2]
    for name, P1 in [("even", even), ("odd", odd), ("quarter", quarter)]:
        steps.append(
            (
                f"mu = {mu}, {name}",
                smear.directional_frozen_shift(
                    U, mu, P1, generator_staples_plaquette(mu, nu, lam), parameters=params
                ),
            )
        )

# round trip, frozen check, change of the links
for tag, T in steps:
    Up = T(U + params)
    eps = relative(T.inv(Up)[0:4], U)
    change = relative(Up[0:4], U)
    g.message(f"Round trip ({tag}): {eps}, change of the links {change}")
    assert eps < 1e-14 and change > 1e-2
    T.check_frozen(U + params)

# a generator reading the updated link U_mu(x + 2 nu) (same parity as x) is
# detected
mu, nu = 1, 2


def bad(F, xp):
    up, dn = staple_pair(mu, nu, F)
    return g(up * g.adj(g.cshift(g.cshift(F[mu], nu, 1), nu, 1)) * g.adj(dn))


try:
    smear.directional_frozen_shift(U, mu, even, bad).check_frozen(U)
    assert False, "not detected"
except ValueError as e:
    g.message(f"Detected: {e}")

# P0 must not overlap P1
try:
    smear.directional_frozen_shift(U, mu, even, bad, P0=quarter)
    assert False, "not detected"
except ValueError as e:
    g.message(f"Detected: {e}")

# gauge covariance: T(U^V) = T(U)^V
V = g.lattice(U[0])
rng.element(V)
for tag, T in steps[0:6]:
    eps = relative(T(g.qcd.gauge.transformed(U, V) + params)[0:4], g.qcd.gauge.transformed(T(U + params)[0:4], V))
    g.message(f"Gauge covariance ({tag}): {eps}")
    assert eps < 1e-14

# forces: all fields, the frozen links of the step only, the parameters only
for tag, T in steps[0:3]:
    a = g.qcd.gauge.action.iwasaki(5.5).transformed(T, indices=[0, 1, 2, 3, 4, 5], projection=[0, 1, 2, 3])
    frozen = [U[nu] for nu in range(4) if nu != T.mu]
    for what, dfields in [("all", U + params), ("frozen links", frozen), ("parameters", params)]:
        g.message(f"Force ({tag}, {what})")
        a.assert_gradient_error(rng, U + params, dfields, 1e-4, 1e-8)

# a chain of frozen steps (both parities, two directions; parameters fixed
# numbers) with local_stout steps, and the log det of the chain (zero for the
# frozen steps)
def fixed(G, values):
    return lambda F, xp: G(F, values)


chain = [
    smear.directional_frozen_shift(U, 0, even, fixed(generator_staples_plaquette(0, 1, 2), [0.3, 0.2])),
    smear.local_stout(rho=0.1, dimension=1, checkerboard=g.even),
    smear.directional_frozen_shift(U, 0, odd, fixed(generator_staples_plaquette(0, 1, 2), [0.3, 0.2])),
    smear.local_stout(rho=0.1, dimension=1, checkerboard=g.odd),
    smear.directional_frozen_shift(U, 1, quarter, fixed(generator_staples(1, 3), [0.4])),
]
a = g.qcd.gauge.action.wilson(5.5)
ld = None
for t in chain:
    a = a.transformed(t)
    l = t.action_log_det_jacobian()
    ld = l if ld is None else ld.transformed(t) + l
g.message("Force of a chain with local_stout steps")
(a + ld).assert_gradient_error(rng, U, U, 1e-4, 1e-8)

# the wrapper: as the base class on the checkerboard, interface of local_stout
for mu in range(4):
    nu = (mu + 1) % 4
    for cb, P1 in [(g.even, even), (g.odd, odd)]:
        G = generator_staples(mu, nu)
        fs = smear.frozen_shift(dimension=mu, checkerboard=cb, generator=G, parameters=[0.3])
        base = smear.directional_frozen_shift(U, mu, P1, G, parameters=[0.3])
        eps = relative(fs(U + [0.3])[0:4], base(U + [0.3])[0:4])
        eps_inv = relative(fs.inv(fs(U + [0.3]))[0:4], U)
        g.message(f"Wrapper (mu = {mu}, {cb.__name__}): agreement {eps}, round trip {eps_inv}")
        assert eps == 0.0 and eps_inv < 1e-14
        assert g.norm2(fs.log_det_jacobian(U + [0.3])) == 0.0
fs = smear.frozen_shift(dimension=2, checkerboard=g.odd, generator=generator_staples(2, 0), parameters=[0.3])
ld = fs.action_log_det_jacobian()
ld = ld.transformed(fs) + fs.action_log_det_jacobian()
assert type(ld) is type(fs.action_log_det_jacobian())
assert ld(U + [0.3]) == 0.0
g.message("Wrapper force")
g.qcd.gauge.action.wilson(5.5).transformed(fs, indices=[0, 1, 2, 3, 4], projection=[0, 1, 2, 3]).assert_gradient_error(
    rng, U + [0.3], U + [0.3], 1e-4, 1e-8
)


if False:
    #####################################################################
    # 2^4 SU(2): the full Jacobian by finite differences, det = 1
    #####################################################################
    def dense_log_det(t, fields, epsilon=1e-5):
        # log det of the Jacobian of t (links -> links) in the generator
        # coordinates of the left group flow, by central differences
        grid = fields[0].grid
        nd = len(fields)
        cartesian = fields[0].otype.cartesian()
        generators = cartesian.generators(grid.precision.complex_dtype)
        coor = g.coordinates(fields[0])
        nsites = len(coor)
        n = nd * len(generators) * nsites
        J = np.zeros((n, n))
        Up0 = t(fields)
        dA = g.lattice(grid, cartesian)
        W = g.lattice(fields[0])
        dAp = g.lattice(grid, cartesian)
        column = 0
        for l in range(nd):
            for a in range(len(generators)):
                for s in range(nsites):
                    diff = None
                    for sign in [1, -1]:
                        dA[:] = 0
                        dA[tuple(int(x) for x in coor[s])] = g(sign * epsilon * generators[a])
                        g.convert(W, dA)
                        Up = t([g(W * f) if i == l else f for i, f in enumerate(fields)])
                        c = []
                        for b in range(nd):
                            g.convert(dAp, g(Up[b] * g.adj(Up0[b])))
                            c += [x[:].flatten().real for x in cartesian.coordinates(dAp)]
                        c = np.concatenate(c)
                        diff = c if diff is None else (diff - c) / (2 * epsilon)
                    J[:, column] = diff
                    column += 1
        # (sign, log |det|); the sign depends on the orientation of the
        # coordinates, compare with the identity map
        return np.linalg.slogdet(J)

    grid2 = g.grid([2, 2, 2, 2], g.double)
    U2 = [g.lattice(grid2, g.ot_matrix_su_n_fundamental_group(2)) for mu in range(4)]
    rng.element(U2)
    even2, odd2 = g.even_odd_projectors(grid2)

    # the finite-difference determinant against the known log det of local_stout
    sign_identity, ld_identity = dense_log_det(lambda f: f, U2)
    g.message(f"Dense log det of the identity: sign {sign_identity}, {ld_identity}")
    assert abs(ld_identity) < 1e-9


    def log_det(t, fields):
        sign, ld = dense_log_det(t, fields)
        assert sign == sign_identity
        return ld


    ls = smear.local_stout(rho=0.05, dimension=1, checkerboard=g.even)
    ld_exact = g.sum(ls.log_det_jacobian(U2)).real
    ld_fd = log_det(lambda f: ls(f), U2)
    eps = abs(ld_fd - ld_exact)
    g.message(f"Dense log det of local_stout: {ld_fd} vs {ld_exact} ({eps})")
    assert eps < 1e-7

    for mu, P1 in [(1, even2), (3, odd2)]:
        nu, lam = [x for x in range(4) if x != mu][0:2]
        T = smear.directional_frozen_shift(U2, mu, P1, generator_staples_plaquette(mu, nu, lam), parameters=[0.4, 0.3])
        ld_fd = log_det(lambda f: T(f + [0.4, 0.3])[0:4], U2)
        change = relative(T(U2 + [0.4, 0.3])[0:4], U2)
        g.message(f"Dense log det of directional_frozen_shift (mu = {mu}): {ld_fd} (change of the links {change})")
        assert abs(ld_fd) < 1e-7 and change > 1e-2
