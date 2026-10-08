#!/usr/bin/env python3
#
# AD of nodes that encapsulate a list of fields.
#
# The motivating case is a gauge field U = [U0, U1, U2, U3] (a list of four
# link lattices).  rad.node(U) wraps the whole list in a single list node;
# individual links are accessed with nU[i] (the node's __getitem__), and the
# gradient flows back into the list element-wise (converted to cartesian, as
# for a plain gauge field).
#
# Convention with the functional machinery: a gauge field is ONE field that
# happens to be a list, so it is passed wrapped in a one-element list --
# f.gradient([U], [U]) and assert_gradient_error(rng, [U], [U], ...).
#
import gpt as g
import numpy as np

rad = g.ad.reverse

grid = g.grid([4, 4, 4, 4], g.double)
rng = g.random("list_node")
U = g.qcd.gauge.random(grid, rng)


def n2(x):
    return float(g.inner_product(x, x).real)


#####################################
# 1. construction + element access
#####################################
nU = rad.node(U)
g.message(f"list node container: {nU._container}")
assert nU._container.tag[0] is list
assert len(nU.value) == len(U)
assert nU.otype == U[0].otype  # element otype
assert nU.grid is U[0].grid  # element grid

# nU[i] is a node holding the i-th link; its forward value is U[i]
for i in range(len(U)):
    li = nU[i]
    assert isinstance(li, g.ad.reverse.node_base)
    v = li(with_gradients=False)
    err = n2(v - U[i])
    assert err == 0.0
g.message("list node construction + __getitem__ forward values: OK")


#####################################
# 2. gradient routes to the right list element
#####################################
# a graph that only reads link 0: its gradient must land on element 0 and
# leave the other elements exactly zero
nU = rad.node(U)
S = g.sum(g.trace(nU[0] * nU[0])).real
S()
norms = [n2(x) for x in nU.gradient]
g.message(f"single-link gradient norms: {norms}")
assert norms[0] > 0.0
for i in range(1, len(U)):
    assert norms[i] == 0.0
# every element is converted to the cartesian (algebra) representation
for i in range(len(U)):
    assert "algebra" in nU.gradient[i].otype.__name__
# and link 0's gradient matches the one from a dedicated per-link node
n0ref = rad.node(g.copy(U[0]))
Sref = g.sum(g.trace(n0ref * n0ref)).real
Sref()
d = n2(nU.gradient[0] - n0ref.gradient)
g.message(f"link-0 gradient vs per-link node: {d}")
assert d < 1e-22
g.message("list node gradient routing + cartesian conversion: OK")


#####################################
# 3. first derivative of a plaquette built from list nodes
#####################################
# P(x) = U0(x) U1(x+e0) U0(x+e1)^dagger U1(x)^dagger
def plaq(V):
    U0, U1 = V[0], V[1]
    return U0 * g.cshift(U1, 0, 1) * g.adj(g.cshift(U0, 1, 1)) * g.adj(U1)


# plain reference
Pref = g.sum(g.trace(plaq(U))).real

# list-node graph
nU = rad.node(U)
P = g.sum(g.trace(plaq(nU))).real
pval = P(with_gradients=False)
eps = abs(float(pval) - float(Pref))
g.message(f"plaquette (list node) forward: {pval} versus {Pref}: {eps}")
assert eps < 1e-12

f = P.functional(nU)
f.assert_gradient_error(rng, [U], [U], 1e-3, 1e-8)
g.message("list node plaquette: forward + gradient + cartesian: OK")


#####################################
# 4. second derivative (HVP) of a gauge action
#####################################
# A k-th derivative is obtained with k reverse passes, all but the last
# recorded (create_graph=True), as in tests/ad/higher_order.py.  After
# action(nU).backward(create_graph=True) the 1st derivative nU.gradient is a
# LIST of node graphs (one per link) over the same leaf; the backward of its
# contraction with a plain direction deposits the HVP into nU.gradient.
#
# The list-node result is cross-checked against per-link leaves
# ([node(u) for u in Ug]) and against the Taylor identity.
gridg = g.grid([4, 4, 4, 4], g.double)
rngg = g.random("list_node_gauge")
Ug = g.qcd.gauge.random(gridg, rngg, scale=2.0)
action = g.qcd.gauge.action.differentiable_iwasaki(2.95)
Nd = len(Ug)


def contract(a, b):
    return sum(g.group.inner_product(a[mu], b[mu]) for mu in range(Nd))


dA = rngg.normal_element(g.group.cartesian(Ug))

# list node
nU = rad.node(Ug)
action(nU).backward(create_graph=True)
contract(nU.gradient, dA).backward()
H_list = [g(x) for x in nU.gradient]

# per-link nodes
nUg = [rad.node(u) for u in Ug]
action(nUg).backward(create_graph=True)
contract([x.gradient for x in nUg], dA).backward()
H_link = [g(nUg[mu].gradient) for mu in range(Nd)]

diff = max(n2(H_list[mu] - H_link[mu]) for mu in range(Nd))
g.message(f"HVP list node vs per-link nodes: {diff}")
assert diff < 1e-16

# Taylor identity: S(U + eps A) = S + eps IP(F,A) + eps^2/2 IP(A,H) + O(eps^3)
F = g.qcd.gauge.action.iwasaki(2.95).gradient(Ug, Ug)
a0 = g(action(Ug))
eps = 1e-4
Ue = [g(g.group.compose(g(eps * dA[mu]), Ug[mu])) for mu in range(Nd)]
a1 = g(action(Ue))
F_A = sum(g.group.inner_product(F[mu], dA[mu]) for mu in range(Nd))
A_H = sum(g.group.inner_product(dA[mu], H_list[mu]) for mu in range(Nd))
res = abs(a1 - a0 - eps * F_A - 0.5 * eps**2 * A_H) / (eps**2 * (abs(A_H) + 1e-30))
g.message(f"HVP Taylor 2nd-order relative residual: {res}")
assert res < 1e-3
g.message("list node second derivative (HVP): OK")


#####################################
# 5. third derivative of a gauge action
#####################################
# Three reverse passes, the first two recorded: the backward of the
# contraction with dA deposits the (recorded) HVP into nU.gradient; the
# backward of its contraction with dB deposits the 3rd derivative.
# Cross-checked against per-link leaves (which higher_order.py validates
# against finite differences).
dB = g.random("list_node_gauge2").normal_element(g.group.cartesian(Ug))

# list node
nU = rad.node(Ug)
action(nU).backward(create_graph=True)
contract(nU.gradient, dA).backward(create_graph=True)
contract(nU.gradient, dB).backward()
G_list = [g(x) for x in nU.gradient]

# per-link nodes
nUg = [rad.node(u) for u in Ug]
action(nUg).backward(create_graph=True)
contract([x.gradient for x in nUg], dA).backward(create_graph=True)
contract([x.gradient for x in nUg], dB).backward()
G_link = [g(nUg[mu].gradient) for mu in range(Nd)]

diff = max(n2(G_list[mu] - G_link[mu]) for mu in range(Nd))
g.message(f"3rd derivative list node vs per-link nodes: {diff}")
assert diff < 1e-16
g.message("list node third derivative: OK")
