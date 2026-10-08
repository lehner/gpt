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
# links as one node), or a plain lattice (a constant).  The inputs
# occupy fields m..n-1.
#
# The forward is a single kernel pass computing all m outputs.  The backward
# runs the ADJOINT of the stencil code, which is a stencil in closed form
# (the product rule per factor, the m output flows as extra inputs).  The
# adjoint reads no flow slot, so the backward is also a single kernel pass.
#
# The flagship case is the fused two-output plaquette (P and P^dagger) as
# stencil(nP_listnode, nU_listnode).  Every derivative order is cross-checked
# against the equivalent per-link-node inputs, and the 1st and 2nd derivatives
# against finite differences.
#
# The consumers of the stencil output are deliberately NONLINEAR (quadratic in
# the outputs).  With a linear consumer the flow that reaches the output does
# not depend on the input links at all, so the d(flow)/dU half of every
# derivative beyond the first is never exercised: a stencil whose recorded
# reverse pass drops that dependence still reproduces every cross-check here
# exactly.  The cross-checks compare two ways of SPECIFYING the same
# computation, so both sides run through this foundation and a defect they
# share is invisible -- hence the finite-difference reference for the 2nd
# derivative as well.
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
    return float(g.norm2(x))


def act_fused(a, b):
    # the action consumed from the fused two-output stencil, NONLINEAR in the
    # outputs (see the header): the same combination the linear version used,
    # applied to their SQUARES.  (tr(X X) with X = P + i P^dagger is not
    # usable: target 1 is the exact pointwise adjoint of target 0, so the two
    # squares cancel and the action vanishes identically.)
    return 2 * g.sum(g.trace(a * a + 1j * (b * b))).real


def act_single(a):
    # the same, for a single-output stencil: tr(P P)
    return 2 * g.sum(g.trace(a * a)).real


def flowed_links(t, dA):
    # U(t) = exp(t dA) U, the group flow the finite differences run along
    return [g(g.group.compose(g(t * dA[mu]), U[mu])) for mu in range(Nd)]


def contract(a, b):
    return sum(g.group.inner_product(a[mu], b[mu]) for mu in range(Nd))


def hvp_fd(build, dA, eps=1e-4):
    # <dA, dS/dU>(U(t)) differentiated by a central difference at t = 0, i.e.
    # an independent reference for the contracted 2nd derivative.  The 1st
    # derivative it differentiates is itself checked against finite
    # differences by assert_gradient_error.
    def F(t):
        n = [rad.node(u) for u in flowed_links(t, dA)]
        build(n)()
        return contract(dA, [g(n[mu].gradient) for mu in range(Nd)])

    return (F(eps) - F(-eps)) / (2.0 * eps)


def assert_hvp_vs_fd(H, build, dA, tag, tol=1e-6):
    # H[mu] = d/dU_mu <dA, dS/dU>, as produced by a recorded reverse pass
    q_ad = contract(dA, H)
    q_fd = hvp_fd(build, dA)
    rel = abs(q_ad - q_fd) / abs(q_fd)
    g.message(f"2nd deriv vs finite differences ({tag}): {q_ad} versus {q_fd}: {rel}")
    assert rel < tol


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
S_plain = act_fused(Ps0, Ps1)
p0 = 2 * g.sum(g.trace(Ps0)).real / gsites / 4 / 3 / 3
p1 = 2 * g.sum(g.trace(Ps1)).real / gsites / 4 / 3 / 3
g.message(f"fused P: {p0}, fused P^dag: {p1}, reference: {Pref}")
assert abs(float(p0) - float(Pref)) < 1e-14
assert abs(float(p1) - float(Pref)) < 1e-14

# node fused: list-node output (2) + list-node input (4)
nP = rad.node([Ps0, Ps1])
nU = rad.node(U)
stencil(nP, nU)
S = act_fused(nP[0], nP[1])
pval = S(with_gradients=False)
# relative: the nonlinear action is O(volume), not O(1)
eps = abs(float(pval) - float(S_plain)) / abs(float(S_plain))
g.message(f"fused forward node: {pval} versus {S_plain}: {eps}")
assert eps < 1e-12

# 1st derivative: finite differences + fused list input vs per-link inputs
f = S.functional(nU)
f.assert_gradient_error(rng, [U], [U], 1e-3, 1e-8)
# the adjoint is a single compiled kernel (one backward pass), compiled at
# the first backward; the cache key is (output count, flowed inputs)
assert any(k[0] == 2 and v[1] is not None for k, v in stencil._node_adj.items())
nPg = rad.node([g.copy(Ps0), g.copy(Ps1)])
nUg = [rad.node(u) for u in U]
stencil(nPg, *nUg)
T = act_fused(nPg[0], nPg[1])
T()
diff = max(n2(nU.gradient[mu] - nUg[mu].gradient) for mu in range(Nd))
g.message(f"1st deriv: fused list input vs per-link inputs: {diff}")
assert diff < 1e-16
g.message("fused plaquette 1st derivative: OK")

# 2nd derivative (HVP): the reverse pass of S is recorded (create_graph=True),
# nU2.gradient is a node graph over the same leaf; the backward of its
# contraction with the plain direction dA deposits d/dU <dA, dS/dU>
dA = rng.normal_element(g.group.cartesian(U))
nP2 = rad.node([g.copy(Ps0), g.copy(Ps1)])
nU2 = rad.node(U)
stencil(nP2, nU2)
act_fused(nP2[0], nP2[1]).backward(create_graph=True)
# the recorded 1st-derivative slots (nU2.gradient) are ADJOINT stencil nodes
# -- a different-code stencil acting on nodes again -- not cshift/mul
# expressions; the recursion is a tower of stencils with no cshift
slot_str = str(nU2.gradient[0])
assert "stencil" in slot_str and "cshift" not in slot_str, (
    "recorded slot should be a stencil node, not a cshift expression")
g.message("2nd deriv: recorded slots are stencil nodes (no cshift)")
contract(nU2.gradient, dA).backward()
H_list = [g(x) for x in nU2.gradient]
nP2g = rad.node([g.copy(Ps0), g.copy(Ps1)])
nU2g = [rad.node(u) for u in U]
stencil(nP2g, *nU2g)
act_fused(nP2g[0], nP2g[1]).backward(create_graph=True)
contract([x.gradient for x in nU2g], dA).backward()
H_link = [g(nU2g[mu].gradient) for mu in range(Nd)]
diff = max(n2(H_list[mu] - H_link[mu]) for mu in range(Nd))
g.message(f"2nd deriv (HVP): fused list input vs per-link inputs: {diff}")
assert diff < 1e-16


def build_fused(n):
    # the fused two-output action over four per-link nodes
    out = rad.node([g.lattice(grid, U[0].otype) for _ in range(2)])
    stencil(out, *n)
    return act_fused(out[0], out[1])


assert_hvp_vs_fd(H_list, build_fused, dA, "fused plaquette")
g.message("fused plaquette 2nd derivative: OK")

# 3rd derivative: two recorded passes, the backward of the contraction with
# dA deposits the (recorded) HVP, its contraction with dB the 3rd derivative
dB = g.random("stencil_3rd").normal_element(g.group.cartesian(U))
nP3 = rad.node([g.copy(Ps0), g.copy(Ps1)])
nU3 = rad.node(U)
stencil(nP3, nU3)
act_fused(nP3[0], nP3[1]).backward(create_graph=True)
contract(nU3.gradient, dA).backward(create_graph=True)
contract(nU3.gradient, dB).backward()
G_list = [g(x) for x in nU3.gradient]
nP3g = rad.node([g.copy(Ps0), g.copy(Ps1)])
nU3g = [rad.node(u) for u in U]
stencil(nP3g, *nU3g)
act_fused(nP3g[0], nP3g[1]).backward(create_graph=True)
contract([x.gradient for x in nU3g], dA).backward(create_graph=True)
contract([x.gradient for x in nU3g], dB).backward()
G_link = [g(nU3g[mu].gradient) for mu in range(Nd)]
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
S1 = act_single(nP1)
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
# Every derivative order is cross-checked against the equivalent hand-written
# single-output plaquette (stencil1): same computation, different code
# specification -> bit-identical.
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
S = act_single(nT)
S()
f = S.functional(*nU)
# 4 link-node arguments -> fields/dfields are the 4-link list U (not [U])
f.assert_gradient_error(rng, U, U, 1e-3, 1e-8)
# cross-check against the hand-written single-output plaquette (same
# computation, different code specification -> gradients must be identical)
nP1b = rad.node(g.copy(P0))
nU1b = [rad.node(u) for u in U]
stencil1(nP1b, *nU1b)
S1b = act_single(nP1b)
S1b()
diff = max(n2(nU1b[mu].gradient - nU[mu].gradient) for mu in range(Nd))
g.message(f"path vs hand-written single-output plaquette 1st deriv: {diff}")
assert diff < 1e-16
g.message("path-based plaquette 1st derivative: OK")

# 2nd derivative (HVP): path-based vs hand-written single-output plaquette
# (dA is defined in the fused-plaquette sections above)
# path-based
nU2 = [rad.node(u) for u in U]
act_single(ptm(nU2)).backward(create_graph=True)
contract([x.gradient for x in nU2], dA).backward()
H_path = [g(nU2[mu].gradient) for mu in range(Nd)]
# hand-written
nU21 = [rad.node(u) for u in U]
T1 = rad.node(g.copy(P0))
stencil1(T1, *nU21)
act_single(T1).backward(create_graph=True)
contract([x.gradient for x in nU21], dA).backward()
H_hw = [g(nU21[mu].gradient) for mu in range(Nd)]
diff = max(n2(H_path[mu] - H_hw[mu]) for mu in range(Nd))
g.message(f"path vs hand-written 2nd deriv (HVP): {diff}")
assert diff < 1e-16


def build_path(n):
    # the single-output action over four per-link nodes, with the
    # target allocated by parallel_transport_matrix.__call__
    return act_single(ptm(n))


assert_hvp_vs_fd(H_path, build_path, dA, "path-based plaquette")
g.message("path-based plaquette 2nd derivative: OK")

# 3rd derivative: path-based vs hand-written single-output plaquette
# path-based
nU3 = [rad.node(u) for u in U]
act_single(ptm(nU3)).backward(create_graph=True)
contract([x.gradient for x in nU3], dA).backward(create_graph=True)
contract([x.gradient for x in nU3], dB).backward()
G_path = [g(nU3[mu].gradient) for mu in range(Nd)]
# hand-written
nU31 = [rad.node(u) for u in U]
T3b = rad.node(g.copy(P0))
stencil1(T3b, *nU31)
act_single(T3b).backward(create_graph=True)
contract([x.gradient for x in nU31], dA).backward(create_graph=True)
contract([x.gradient for x in nU31], dB).backward()
G_hw = [g(nU31[mu].gradient) for mu in range(Nd)]
diff = max(n2(G_path[mu] - G_hw[mu]) for mu in range(Nd))
g.message(f"path vs hand-written 3rd deriv: {diff}")
assert diff < 1e-16
g.message("path-based plaquette 3rd derivative: OK")



#####################################
# local temporaries (kernel-owned per-site fields) in node mode
#####################################
# The improved gauge action as ONE stencil whose up/down staples are local
# temporaries shared by the plaquette and rectangle terms
# (g.qcd.gauge.action.staple_stencil).  Its adjoint is two stencils (stage A
# with local temporaries, stage B without), so every derivative order stays
# a stencil.  Cross-checked against the cshift node graph of
# differentiable_improved_with_rectangle at 1st, 2nd and 3rd order.
from gpt.qcd.gauge.action.staple_stencil import staple_stencil_action

beta, c1 = 2.95, -0.331
act_st = staple_stencil_action(beta, 1.0 - 8.0 * c1, c1)
act_cs = g.qcd.gauge.action.differentiable_improved_with_rectangle(beta, c1)

v_st, v_cs = act_st(U), g(act_cs(U))
eps = abs(v_st - complex(v_cs).real) / abs(v_st)
g.message(f"local temporaries: action value vs cshift graph: {eps}")
assert eps < 1e-14

# 1st derivative (functional force), and with_value=False gives the same
# gradient without computing the root value
nU = [rad.node(g.copy(u)) for u in U]
S_st = act_st(nU)
F_st = S_st.functional(*nU).gradient(U, U)
nU2 = [rad.node(g.copy(u)) for u in U]
F_cs = act_cs(nU2).functional(*nU2).gradient(U, U)
diff = max(n2(F_st[mu] - F_cs[mu]) / n2(F_cs[mu]) for mu in range(Nd))
g.message(f"local temporaries: force vs cshift graph: {diff}")
assert diff < 1e-28
nU3 = [rad.node(g.copy(u)) for u in U]
S3n = act_st(nU3)
assert S3n(with_value=False) is None
diff = max(n2(nU3[mu].gradient - F_st[mu]) / n2(F_st[mu]) for mu in range(Nd))
g.message(f"with_value=False: gradient vs functional force: {diff}")
assert diff < 1e-28
act_st.gradient(U, U)
g.qcd.gauge.action.iwasaki(beta).assert_gradient_error(rng, U, U, 1e-3, 1e-8)


def hvp_of(action, d):
    nU = [rad.node(u) for u in U]
    action(nU).backward(create_graph=True)
    contract([x.gradient for x in nU], d).backward()
    return [g(nU[mu].gradient) for mu in range(Nd)]


H_st = hvp_of(act_st, dA)
H_cs = hvp_of(act_cs, dA)
diff = max(n2(H_st[mu] - H_cs[mu]) / n2(H_cs[mu]) for mu in range(Nd))
g.message(f"local temporaries: HVP vs cshift graph: {diff}")
assert diff < 1e-28


def d3_of(action):
    nU = [rad.node(u) for u in U]
    action(nU).backward(create_graph=True)
    contract([x.gradient for x in nU], dA).backward(create_graph=True)
    contract([x.gradient for x in nU], dB).backward()
    return [g(nU[mu].gradient) for mu in range(Nd)]

G_st = d3_of(act_st)
G_cs = d3_of(act_cs)
diff = max(n2(G_st[mu] - G_cs[mu]) / n2(G_cs[mu]) for mu in range(Nd))
g.message(f"local temporaries: 3rd derivative vs cshift graph: {diff}")
assert diff < 1e-28
g.message("local temporaries in node mode: OK")

# a stencil with a single input field (all factors at the zero point): the
# cubic f(X) = X + a X^2 + b X^3 of a general complex matrix field (the
# word sums of g.ml.layer.word_sum).  The adjoint of a one-input stencil has a
# single flow slot, a one-element list node in recorded passes.  HVP and 3rd
# derivative of the nonlinear Re sum tr f(X) f(X) against the node graph
X = g.mcolor(grid)
rng.cnormal(X, sigma=0.3)
DX, DY = g.mcolor(grid), g.mcolor(grid)
rng.cnormal([DX, DY])
ca, cb = 0.3 - 0.1j, -0.2 + 0.05j
zero = (0,) * Nd
cubic = g.stencil.matrix(X, [zero], [(0, -1, 1.0, [(1, 0, 0)]), (0, 0, ca, [(1, 0, 0)] * 2), (0, 0, cb, [(1, 0, 0)] * 3)])


def cubic_st(x):
    out = g.lattice(X)
    if isinstance(x, rad.node_base):
        out = rad.node(out)
    cubic(out, x)
    return out


def cubic_graph(x):
    x2 = x * x
    return x + x2 * ca + x2 * x * cb


def act_cubic(f, x):
    y = f(x)
    return g.sum(g.trace(y * y)).real


def leaf(x):
    return rad.node(x, infinitesimal_to_cartesian=False)


def contract_dir(grad, d):
    # Re sum tr(d^dag grad), d a plain direction
    return g.sum(g.trace(g.adj(d) * grad)).real


def derivatives_cubic(f):
    # gradient, HVP along DX, 3rd derivative along DX, DY
    n1 = leaf(X)
    act_cubic(f, n1).backward()
    G = g(n1.gradient)
    n2_ = leaf(X)
    act_cubic(f, n2_).backward(create_graph=True)
    contract_dir(n2_.gradient, DX).backward()
    H = g(n2_.gradient)
    n3 = leaf(X)
    act_cubic(f, n3).backward(create_graph=True)
    contract_dir(n3.gradient, DX).backward(create_graph=True)
    contract_dir(n3.gradient, DY).backward()
    D3 = g(n3.gradient)
    return G, H, D3

plain = cubic_st(X)
eps = n2(plain - g(cubic_graph(X))) / n2(plain)
g.message(f"single-input stencil: value vs graph {eps}")
assert eps < 1e-28
for name, a, b in zip(["gradient", "HVP", "3rd derivative"], derivatives_cubic(cubic_st), derivatives_cubic(cubic_graph)):
    eps = n2(a - b) / n2(b)
    g.message(f"single-input stencil: {name} vs graph {eps}")
    assert eps < 1e-26

# constant inputs: the adjoint computes only the flows of gradient-carrying
# inputs (the other slots are dropped).  f(X; C, D) = X + C X X + X D X with C
# a plain constant and D a node without gradient, against the node graph at
# 1st to 3rd order
C = g.mcolor(grid)
D = g.mcolor(grid)
rng.cnormal([C, D], sigma=0.3)
mixed = g.stencil.matrix(
    X,
    [zero],
    [(0, -1, 1.0, [(1, 0, 0)]), (0, 0, 1.0, [(2, 0, 0), (1, 0, 0), (1, 0, 0)]), (0, 0, ca, [(1, 0, 0), (3, 0, 0), (1, 0, 0)])],
)


def mixed_st(x):
    out, d = g.lattice(X), D
    if isinstance(x, rad.node_base):
        out = rad.node(out)
        d = rad.node(d, with_gradient=False)
    mixed(out, x, C, d)
    return out


def mixed_graph(x):
    c, d = C, D
    if isinstance(x, rad.node_base):
        c = rad.node(c, with_gradient=False)
        d = rad.node(d, with_gradient=False)
    return x + c * x * x + x * d * x * ca

before = len(mixed._node_adj) if hasattr(mixed, "_node_adj") else 0
for name, a, b in zip(["gradient", "HVP", "3rd derivative"], derivatives_cubic(mixed_st), derivatives_cubic(mixed_graph)):
    eps = n2(a - b) / n2(b)
    g.message(f"constant stencil inputs: {name} vs graph {eps}")
    assert eps < 1e-26
# the adjoint of the first level has a single flow slot (X)
flowed = [k[1] for k in mixed._node_adj if k[0] == 1]
g.message(f"constant stencil inputs: flowed inputs of the cached adjoints {flowed}")
assert (1,) in flowed
