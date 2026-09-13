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
assert eps < 1e-14

f = P.functional(nU)
f.assert_gradient_error(rng, [U], [U], 1e-3, 1e-8)
g.message("list node plaquette: forward + gradient + cartesian: OK")
