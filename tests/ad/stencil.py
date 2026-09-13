#!/usr/bin/env python3
#
# AD of compiled matrix stencils that map a list of input fields to a (list
# of) output field(s), with both sides represented as nodes:
#
#     stencil(output, *inputs)
#
# `output` is field(s) 0..m-1: a single node (one output) or a LIST node (a
# fused kernel with several outputs).  Each input argument is a single node,
# a LIST node (expanding to one input field per element, e.g. the 4 gauge
# links as one node), or a plain lattice (a constant/temp).  The inputs
# occupy fields m..n-1.
#
# The forward is a single kernel pass computing all m outputs.  The backward
# runs the ADJOINT of the stencil code, which is a stencil in closed form
# (the product rule per factor, the m output flows as extra inputs).  For a
# temp-free code the adjoint reads no flow slot, so it fuses to a single
# stage: the backward is also a single kernel pass.
#
# The flagship case is the fused two-output plaquette (P and P^dagger) as
# stencil(nP_listnode, nU_listnode).  Every derivative order is cross-checked
# against the equivalent per-link-node inputs, and the 1st derivative against
# finite differences.
#
import gpt as g

rad = g.ad.reverse

grid = g.grid([4, 4, 4, 8], g.double)
rng = g.random("stencil")
U = g.qcd.gauge.random(grid, rng)
Pref = g.qcd.gauge.plaquette(U)
Nd = len(U)
gsites = U[0].grid.gsites
# points / field conventions: the output(s) are field 0..m-1, the links are
# the following fields, the shifts are single-direction points
_P = 0
_U = [2, 3, 4, 5]
_Sp = [1, 2, 3, 4]
pts = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]


def n2(x):
    return float(g.inner_product(x, x).real)


def list_dir(dA, depth):
    # a `depth`-deep list node holding the direction dA (plain links)
    cU = g.group.cartesian(U)
    for mu in range(Nd):
        cU[mu] @= dA[mu]
    nd = rad.node(cU)
    for _ in range(depth - 1):
        nd = rad.node(nd)
    return nd


def link_dirs(dA, depth):
    # the same direction as `depth`-deep per-link nodes
    cU = [g.group.cartesian(u) for u in U]
    for mu in range(Nd):
        cU[mu] @= dA[mu]
    nds = [rad.node(c) for c in cU]
    for _ in range(depth - 1):
        nds = [rad.node(x) for x in nds]
    return nds


#####################################
# fused two-output plaquette: list node -> list node
#####################################
# P(x)     = U_mu(x) U_nu(x+mu) U_mu(x+nu)^dagger U_nu(x)^dagger   (target 0)
# P^dag(x) = U_nu(x) U_mu(x+nu) U_nu(x+mu)^dagger U_mu(x)^dagger   (target 1)
P0 = g.copy(U[0])
P1 = g.copy(U[0])
Ps0 = g.copy(P0)
Ps1 = g.copy(P1)
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
stencil = g.stencil.matrix(P0, pts, code)

# plain fused forward (one kernel pass computes both outputs)
stencil(Ps0, Ps1, *U)
S_plain = 2 * g.sum(g.trace(Ps0 + 1j * Ps1)).real
p0 = 2 * g.sum(g.trace(Ps0)).real / gsites / 4 / 3 / 3
p1 = 2 * g.sum(g.trace(Ps1)).real / gsites / 4 / 3 / 3
g.message(f"fused P: {p0}, fused P^dag: {p1}, reference: {Pref}")
assert abs(float(p0) - float(Pref)) < 1e-14
assert abs(float(p1) - float(Pref)) < 1e-14

# node fused: list-node output (2) + list-node input (4)
nP = rad.node([Ps0, Ps1])
nU = rad.node(U)
stencil(nP, nU)
S = 2 * g.sum(g.trace(nP[0] + 1j * nP[1])).real
pval = S(with_gradients=False)
eps = abs(float(pval) - float(S_plain))
g.message(f"fused forward node: {pval} versus {S_plain}: {eps}")
assert eps < 1e-12
# temp-free: the adjoint fuses to a single compiled kernel (one backward pass)
nstages = len(stencil._node_adj[(2, ())][1])
g.message(f"adjoint stages (backward passes): {nstages}")
assert nstages == 1

# 1st derivative: finite differences + fused list input vs per-link inputs
f = S.functional(nU)
f.assert_gradient_error(rng, [U], [U], 1e-3, 1e-8)
nPg = rad.node([g.copy(Ps0), g.copy(Ps1)])
nUg = [rad.node(u) for u in U]
stencil(nPg, *nUg)
T = 2 * g.sum(g.trace(nPg[0] + 1j * nPg[1])).real
T()
diff = max(n2(nU.gradient[mu] - nUg[mu].gradient) for mu in range(Nd))
g.message(f"1st deriv: fused list input vs per-link inputs: {diff}")
assert diff < 1e-16
g.message("fused plaquette 1st derivative: OK")

# 2nd derivative (HVP)
dA = rng.normal_element(g.group.cartesian(U))
nnP = rad.node(rad.node([g.copy(Ps0), g.copy(Ps1)]))
nnU = rad.node(rad.node(U))
nA = list_dir(dA, 1)
stencil(nnP, nnU)
2 * g.sum(g.trace(nnP[0] + 1j * nnP[1])).real()
# the 1st-derivative slots (nnU.gradient) are ADJOINT stencil nodes -- a
# different-code stencil acting on nodes again -- not cshift/mul expressions;
# the recursion is a tower of stencils with no cshift
slot_str = str(nnU.gradient[0])
assert "stencil" in slot_str and "cshift" not in slot_str, (
    "nested slot should be a stencil node, not a cshift expression")
g.message("2nd deriv: nested slots are stencil nodes (no cshift)")
c = sum(g.group.inner_product(nnU.gradient[mu], nA[mu]) for mu in range(Nd))
c()
H_list = [g(x) for x in nnU.value.gradient]
nnPg = rad.node(rad.node([g.copy(Ps0), g.copy(Ps1)]))
nnUg = [rad.node(rad.node(u)) for u in U]
nAg = link_dirs(dA, 1)
stencil(nnPg, *nnUg)
2 * g.sum(g.trace(nnPg[0] + 1j * nnPg[1])).real()
c = sum(g.group.inner_product(nnUg[mu].gradient, nAg[mu]) for mu in range(Nd))
c()
H_link = [g(nnUg[mu].value.gradient) for mu in range(Nd)]
diff = max(n2(H_list[mu] - H_link[mu]) for mu in range(Nd))
g.message(f"2nd deriv (HVP): fused list input vs per-link inputs: {diff}")
assert diff < 1e-16
g.message("fused plaquette 2nd derivative: OK")

# 3rd derivative
dB = g.random("stencil_3rd").normal_element(g.group.cartesian(U))
nnnP = rad.node(rad.node(rad.node([g.copy(Ps0), g.copy(Ps1)])))
nnnU = rad.node(rad.node(rad.node(U)))
nA = list_dir(dA, 2)
nB = list_dir(dB, 1)
stencil(nnnP, nnnU)
2 * g.sum(g.trace(nnnP[0] + 1j * nnnP[1])).real()
c = sum(g.group.inner_product(nnnU.gradient[mu], nA[mu]) for mu in range(Nd))
c()
nnU = nnnU.value
c = sum(g.group.inner_product(nnU.gradient[mu], nB[mu]) for mu in range(Nd))
c()
G_list = [g(x) for x in nnU.value.gradient]
nnnPg = rad.node(rad.node(rad.node([g.copy(Ps0), g.copy(Ps1)])))
nnnUg = [rad.node(rad.node(rad.node(u))) for u in U]
nAg = link_dirs(dA, 2)
nBg = link_dirs(dB, 1)
stencil(nnnPg, *nnnUg)
2 * g.sum(g.trace(nnnPg[0] + 1j * nnnPg[1])).real()
c = sum(g.group.inner_product(nnnUg[mu].gradient, nAg[mu]) for mu in range(Nd))
c()
c = sum(g.group.inner_product(nnnUg[mu].value.gradient, nBg[mu]) for mu in range(Nd))
c()
G_link = [g(nnnUg[mu].value.value.gradient) for mu in range(Nd)]
diff = max(n2(G_list[mu] - G_link[mu]) for mu in range(Nd))
g.message(f"3rd deriv: fused list input vs per-link inputs: {diff}")
assert diff < 1e-16
g.message("fused plaquette 3rd derivative: OK")

#####################################
# single-output sub-case: one node output + list node input (m = 1)
#####################################
# the same plaquette, single output (field 0), links at fields 1..4
code1 = []
for mu in range(4):
    for nu in range(mu):
        code1.append(
            {
                "target": 0,
                "accumulate": -1 if len(code1) == 0 else 0,
                "weight": 1.0,
                "factor": [
                    (_U[mu] - 1, _P, 0),
                    (_U[nu] - 1, _Sp[mu], 0),
                    (_U[mu] - 1, _Sp[nu], 1),
                    (_U[nu] - 1, _P, 1),
                ],
            }
        )
stencil1 = g.stencil.matrix(P0, pts, code1)
P1s = g.copy(P0)
stencil1(P1s, *U)
p1v = 2 * g.sum(g.trace(P1s)).real / gsites / 4 / 3 / 3
eps = abs(float(p1v) - float(Pref))
g.message(f"single-output plaquette: {p1v} versus reference {Pref}: {eps}")
assert eps < 1e-14
nP1 = rad.node(P1s)
nU1 = rad.node(U)
stencil1(nP1, nU1)
S1 = 2 * g.sum(g.trace(nP1)).real
S1()
f1 = S1.functional(nU1)
f1.assert_gradient_error(rng, [U], [U], 1e-3, 1e-8)
g.message("single-output plaquette (list input): OK")

#####################################
# performance test
#####################################
f1.gradient([U], [U])
act=g.qcd.gauge.action.wilson(6)
act.gradient(U, U)

t = g.timer("d")
t("AD")
f1.gradient([U], [U])
t("Wilson")
act.gradient(U, U)
t()
g.message(t)

#####################################
# path-based stencil via g.parallel_transport_matrix
#####################################
# the same single-output plaquette as the sub-case above, but specified in
# terms of g.path objects and wrapped by g.parallel_transport_matrix, which
# expands each path into the sequence of single-link factors (and manages the
# point set / field layout itself).  Exercises the
# parallel_transport_matrix -> matrix-stencil -> AD-foundation pipeline.
# Lowest order for now; 2nd/3rd derivatives to follow.
pcode = []
for mu in range(Nd):
    for nu in range(mu):
        pcode.append((0, -1 if len(pcode) == 0 else 0, 1.0,
                      g.path().f(mu).f(nu).b(mu).b(nu)))
ptm = g.parallel_transport_matrix(U, pcode, 1)
# plain forward via the __call__ convention: ptm(U) allocates the target
# itself and returns it (no .stencil extraction)
P0p = ptm(U)
pp = 2 * g.sum(g.trace(P0p)).real / gsites / 4 / 3 / 3
eps = abs(float(pp) - float(Pref))
g.message(f"path-based plaquette: {pp} versus reference {Pref}: {eps}")
assert eps < 1e-14
# node forward + 1st derivative via the __call__ convention: the input link
# nodes are passed and __call__ allocates the target node (resolving the
# stencil call to the AD foundation)
nU = [rad.node(u) for u in U]
nT = ptm(nU)
S = 2 * g.sum(g.trace(nT)).real
S()
f = S.functional(*nU)
# 4 link-node arguments -> fields/dfields are the 4-link list U (not [U])
f.assert_gradient_error(rng, U, U, 1e-3, 1e-8)
# cross-check against the hand-written single-output plaquette (same
# computation, different code specification -> gradients must be identical)
nP1b = rad.node(g.copy(P0))
nU1b = [rad.node(u) for u in U]
stencil1(nP1b, *nU1b)
S1b = 2 * g.sum(g.trace(nP1b)).real
S1b()
diff = max(n2(nU1b[mu].gradient - nU[mu].gradient) for mu in range(Nd))
g.message(f"path vs hand-written single-output plaquette 1st deriv: {diff}")
assert diff < 1e-16
g.message("path-based plaquette 1st derivative: OK")


