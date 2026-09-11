#!/usr/bin/env python3
#
# AD of stencils
#
import gpt as g

rad = g.ad.reverse

grid = g.grid([4, 4, 4, 8], g.double)
rng = g.random("stencil")
U = g.qcd.gauge.random(grid, rng)
Udag = [g(g.adj(u)) for u in U]
P = g.copy(U[0])
Ps = g.copy(P)
Pref = g.qcd.gauge.plaquette(U)

# create a stencil for the plaquette
_P = 0
_U = [1, 2, 3, 4]
_Sp = [1, 2, 3, 4]

code = []
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

stencil_plaquette = g.stencil.matrix(
    P,
    [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)],
    code,
)

stencil_plaquette(Ps, *U)
pval = 2 * g.sum(g.trace(Ps)).real / P.grid.gsites / 4 / 3 / 3

eps = abs(Pref - pval)
g.message(f"Stencil plaquette: {pval} versus reference {Pref}: {eps}")
assert eps < 1e-14

# now run it on nodes
nPs = rad.node(Ps)
nU = [rad.node(u) for u in U]

stencil_plaquette(nPs, *nU)
npval = 2 * g.sum(g.trace(nPs)).real / P.grid.gsites / 4 / 3 / 3
pval = npval(with_gradients=False)
eps = abs(Pref - pval)
g.message(f"Stencil plaquette forward node: {pval} versus reference {Pref}: {eps}")
assert eps < 1e-14

# print graph
g.message(npval)

# now test functional
npval_func = npval.functional(*nU)
npval_func.assert_gradient_error(rng, U, U, 1e-3, 1e-8)

# second derivative
nPs = rad.node(Ps)
nU = [rad.node(u) for u in U]
nnPs = rad.node(nPs)
nnU = [rad.node(u) for u in nU]

stencil_plaquette(nnPs, *nnU)
g.sum(g.trace(nnPs))()

nip = g.inner_product(nnU[0].gradient, nnU[1].gradient)
nup_func = nip.functional(*nU)
nup_func.assert_gradient_error(rng, U, U, 1e-3, 1e-8)


# third derivative
nPs = rad.node(Ps)
nU = [rad.node(u) for u in U]
nnPs = rad.node(nPs)
nnU = [rad.node(u) for u in nU]
nnnPs = rad.node(nnPs)
nnnU = [rad.node(u) for u in nnU]

stencil_plaquette(nnnPs, *nnnU)
g.sum(g.trace(nnnPs))()

nnip = g.inner_product(nnnU[0].gradient, nnnU[1].gradient)
nnip()

nip = g.inner_product(nnU[0].gradient, nnU[1].gradient)
nip()

nnup_func = nip.functional(*nU)
nnup_func.assert_gradient_error(rng, U, U, 1e-3, 1e-8)

