#!/usr/bin/env python3
#
# AD of stencils: a fused stencil computing the plaquette (target 0) and the
# adjoint plaquette (target 1) as two node outputs of one kernel
#
import gpt as g

rad = g.ad.reverse

grid = g.grid([4, 4, 4, 8], g.double)
rng = g.random("stencil")
U = g.qcd.gauge.random(grid, rng)
Udag = [g(g.adj(u)) for u in U]
P = [g.copy(U[0]), g.copy(U[0])]
Ps = [g.copy(P[0]), g.copy(P[1])]
Pref = g.qcd.gauge.plaquette(U)

# create a stencil for the plaquette (target 0) and the adjoint plaquette
# (target 1): P(x) = U_mu(x) U_nu(x+mu) U_mu(x+nu)^dagger U_nu(x)^dagger
# P^dagger(x) = U_nu(x) U_mu(x+nu) U_nu(x+mu)^dagger U_mu(x)^dagger
_P = 0
_U = [2, 3, 4, 5]
_Sp = [1, 2, 3, 4]

code = []
code0 = []
for mu in range(4):
    for nu in range(mu):
        code.append(
            {
                "target": 0,
                "accumulate": -1 if len(code) == 0 else 0,
                "weight": 1.0,
                "factor": [
                    (_U[mu], _P, 0),
                    (_U[nu], _Sp[mu], 0),
                    (_U[mu], _Sp[nu], 1),
                    (_U[nu], _P, 1),
                ],
            }
        )
        code.append(
            {
                "target": 1,
                "accumulate": -1 if len(code) == 1 else 1,
                "weight": 1.0,
                "factor": [
                    (_U[nu], _P, 0),
                    (_U[mu], _Sp[nu], 0),
                    (_U[nu], _Sp[mu], 1),
                    (_U[mu], _P, 1),
                ],
            }
        )
        code0.append(
            {
                "target": 0,
                "accumulate": -1 if len(code0) == 0 else 0,
                "weight": 1.0,
                "factor": [
                    (_U[mu] - 1, _P, 0),
                    (_U[nu] - 1, _Sp[mu], 0),
                    (_U[mu] - 1, _Sp[nu], 1),
                    (_U[nu] - 1, _P, 1),
                ],
            }
        )


stencil_plaquette = g.stencil.matrix(
    P[0],
    [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)],
    code,
)

stencil_plaquette0 = g.stencil.matrix(
    P[0],
    [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)],
    code0,
)

stencil_plaquette(Ps[0], Ps[1], *U)
pval0 = 2 * g.sum(g.trace(Ps[0])).real / P[0].grid.gsites / 4 / 3 / 3
pval1 = 2 * g.sum(g.trace(Ps[1])).real / P[0].grid.gsites / 4 / 3 / 3
eps = abs(Pref - pval0)
g.message(f"Stencil plaquette: {pval0} versus reference {Pref}: {eps}")
assert eps < 1e-14
eps = abs(Pref - pval1)
g.message(f"Stencil adjoint plaquette: {pval1} versus reference {Pref}: {eps}")
assert eps < 1e-14

# now run it on nodes (both targets are node outputs of the fused stencil)
nPs = [rad.node(Ps[0]), rad.node(Ps[1])]
nU = [rad.node(u) for u in U]

stencil_plaquette(nPs[0], nPs[1], *nU)
# the plain reference for the combined scalar: the real part of
# trace(P + i P^dagger) is sum(Re P) - sum(Im P)
pref_val = (
    2 * g.sum(g.trace(Ps[0] + 1j * Ps[1])).real / P[0].grid.gsites / 4 / 3 / 3
)
npval = (
    2 * g.sum(g.trace(nPs[0] + 1j * nPs[1])).real / P[0].grid.gsites / 4 / 3 / 3
)
pval = npval(with_gradients=False)
eps = abs(pref_val - pval)
g.message(f"Stencil plaquette forward node: {pval} versus reference {pref_val}: {eps}")
assert eps < 1e-14

# print graph
g.message(npval)

# now test functional
npval_func = npval.functional(*nU)
act = g.qcd.gauge.action.wilson(6)
t = g.timer("d")
npval_func.gradient(U, U)
act.gradient(U, U)

t("2 x AD")
for _ in range(10):
    npval_func.gradient(U, U)
t("wilson")
for _ in range(10):
    act.gradient(U, U)
t()
g.message(t)

npval_func.assert_gradient_error(rng, U, U, 1e-3, 1e-8)


# second derivative
nPs = [rad.node(Ps[0]), rad.node(Ps[1])]
nU = [rad.node(u) for u in U]
nnPs = [rad.node(p) for p in nPs]
nnU = [rad.node(u) for u in nU]

stencil_plaquette(nnPs[0], nnPs[1], *nnU)
g.sum(g.trace(nnPs[0] + 1j * nnPs[1]))()

nip = g.inner_product(nnU[0].gradient, nnU[1].gradient)
nup_func = nip.functional(*nU)
nup_func.assert_gradient_error(rng, U, U, 1e-3, 1e-8)


# third derivative
nPs = [rad.node(Ps[0]), rad.node(Ps[1])]
nU = [rad.node(u) for u in U]
nnPs = [rad.node(p) for p in nPs]
nnU = [rad.node(u) for u in nU]
nnnPs = [rad.node(p) for p in nnPs]
nnnU = [rad.node(u) for u in nnU]

stencil_plaquette(nnnPs[0], nnnPs[1], *nnnU)
g.sum(g.trace(nnnPs[0] + 1j * nnnPs[1]))()

nnip = g.inner_product(nnnU[0].gradient, nnnU[1].gradient)
nnip()

nip = g.inner_product(nnU[0].gradient, nnU[1].gradient)
nip()

nnup_func = nip.functional(*nU)
nnup_func.assert_gradient_error(rng, U, U, 1e-3, 1e-8)



# now test functional of only the action (fresh nodes: the previous
# section's assert_gradient_error leaves the nU leaf values pointing at
# the last finite-difference composed lattice, not at U)
nPs = rad.node(Ps[0])
nU = [rad.node(u) for u in U]
stencil_plaquette0(nPs, *nU)
# the plain reference for the combined scalar: the real part of
# trace(P + i P^dagger) is sum(Re P) - sum(Im P)
pref_val = (
    2 * g.sum(g.trace(Ps[0])).real / P[0].grid.gsites / 4 / 3 / 3
)
npval = (
    2 * g.sum(g.trace(nPs)).real / P[0].grid.gsites / 4 / 3 / 3
)
pval = npval(with_gradients=False)
eps = abs(pref_val - pval)
g.message(f"Stencil plaquette0 forward node: {pval} versus reference {pref_val}: {eps}")
assert eps < 1e-14

# now test functional
npval_func = npval.functional(*nU)
act = g.qcd.gauge.action.wilson(6)
t = g.timer("d")
npval_func.gradient(U, U)
act.gradient(U, U)

t("AD")
for _ in range(10):
    npval_func.gradient(U, U)
t("wilson")
for _ in range(10):
    act.gradient(U, U)
t()
g.message(t)

npval_func.assert_gradient_error(rng, U, U, 1e-3, 1e-8)
