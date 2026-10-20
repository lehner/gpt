#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2026
#
# g.ad.reverse.checkpoint: values and derivatives (first, second and third
# order) identical to the graph without checkpoints; segments, nesting, an
# order=1 primitive inside, and the smoothers of frozen_generators with
# checkpoint=k in a frozen-shift force and a prescribed-staple log-det force
#
import gpt as g
import numpy as np

rad = g.ad.reverse
rng = g.random("checkpoint")
grid = g.grid([4, 4, 4, 8], g.double)
fg = g.qcd.gauge.smear.frozen_generators


def rel(x, y):
    x, y = g.util.to_list(x), g.util.to_list(y)
    return (sum(g.norm2(p - q) for p, q in zip(x, y)) / sum(g.norm2(q) for q in y)) ** 0.5


#####################################################################
# a complex toy function (conjugate-linear convention, complex seed)
#####################################################################
x0 = g.complex(grid)
rng.cnormal(x0)
a = g.complex(grid)
rng.cnormal(a)


def cubic(X):
    return [g(X[0] * X[0] * X[0] + 0.5 * X[0]), X[0]]


for seed in [1.0, 1.0j]:
    grads = []
    for ckpt in [False, True]:
        n = rad.node(g.copy(x0), infinitesimal_to_cartesian=False)
        y = rad.checkpoint(cubic, [n]) if ckpt else cubic([n])
        c = g.inner_product(a, y[0]) + 0.3 * g.inner_product(y[1], y[0])
        c(initial_gradient=seed)
        grads.append(g.copy(n.gradient))
    eps = rel(grads[1], grads[0])
    g.message(f"toy, seed {seed}: gradient with vs without checkpoint {eps}")
    # (to rounding: the flows of the outputs are summed in another order)
    assert eps < 1e-15

#####################################################################
# a gauge smoothing chain: first, second and third derivatives (to rounding)
#####################################################################
U = g.qcd.gauge.random(grid, rng, scale=0.5)
b = g.lattice(U[0])
rng.element(b)
d = rng.normal_element(g.group.cartesian(U))
d2 = rng.normal_element(g.group.cartesian(U))
smooth = fg.transverse_stout(2, 0.1, 1)


def action(F):
    return sum(g.inner_product(b, F[nu]).real for nu in [0, 1, 3]) + g.norm2(F[0] * F[1])


def chains():
    # (name, chain): the reference without checkpoints first
    yield "none", lambda F: fg.transverse_stout(2, 0.1, 5)(F)
    yield "checkpoint=1", lambda F: fg.transverse_stout(2, 0.1, 5, checkpoint=1)(F)
    yield "checkpoint=2", lambda F: fg.transverse_stout(2, 0.1, 5, checkpoint=2)(F)
    yield "nested", lambda F: rad.checkpoint(
        lambda X: rad.checkpoint(fg.transverse_stout(2, 0.1, 2), fg.transverse_stout(2, 0.1, 3, checkpoint=1)(X)), F
    )


ref = None
for name, chain in chains():
    aU = [rad.node(g.copy(u)) for u in U]
    S = action(chain(aU))
    value = S(with_gradients=False)
    S.backward()
    gradient = [g.copy(x.gradient) for x in aU]
    aU = [rad.node(g.copy(u)) for u in U]
    action(chain(aU)).backward(create_graph=True)
    Q = sum(g.group.inner_product(x, y.gradient) for x, y in zip(d, aU))
    Q.backward(create_graph=True)
    hv = [y.gradient for y in aU]
    R = sum(g.group.inner_product(x, y) for x, y in zip(d2, hv))
    hv_value = [g.copy(y(with_gradients=False)) for y in hv]
    R.backward()
    third = [g.copy(y.gradient) for y in aU]
    del Q, R, hv
    if ref is None:
        ref = (value, gradient, hv_value, third)
        continue
    e = [abs(value - ref[0]) / abs(ref[0]), rel(gradient, ref[1]), rel(hv_value, ref[2]), rel(third, ref[3])]
    g.message(f"chain {name}: value {e[0]}, gradient {e[1]}, Hessian-vector {e[2]}, third derivative {e[3]}")
    assert e[0] < 1e-15 and e[1] < 1e-15 and e[2] < 1e-13 and e[3] < 1e-13

aU = [rad.node(g.copy(u)) for u in U]
g.message("chain checkpoint=2: gradient vs finite differences")
action(fg.transverse_stout(2, 0.1, 5, checkpoint=2)(aU)).functional(*aU).assert_gradient_error(rng, U, U, 1e-4, 1e-8)

#####################################################################
# an order=1 primitive inside f (linear_solve): first order, a recorded
# pass raises
#####################################################################
cg = g.algorithms.inverter.cg(eps=1e-14, maxiter=1000)
green = fg.transverse_green(3, 0.5, cg)
g.default.set_verbose("cg", False)


def green_step(F):
    src = g.qcd.gauge.project.traceless_anti_hermitian(F[0] * g.adj(F[1]))
    w = green(F, src)
    return [g(g.matrix.exp(g(0.1 * w)) * F[0])] + F[1:]


grads = []
for ckpt in [False, True]:
    aU = [rad.node(g.copy(u)) for u in U]
    F = rad.checkpoint(green_step, aU) if ckpt else green_step(aU)
    action(F).backward()
    grads.append([g.copy(x.gradient) for x in aU])
eps = rel(grads[1], grads[0])
g.message(f"linear_solve inside a checkpoint: gradient with vs without {eps}")
assert eps < 1e-12
# (the recorded pass only builds the lazy checkpoint of the vjp; the order=1
# primitive is reached when the second derivative is computed)
aU = [rad.node(g.copy(u)) for u in U]
action(rad.checkpoint(green_step, aU)).backward(create_graph=True)
Q = sum(g.group.inner_product(x, y.gradient) for x, y in zip(d, aU))
try:
    Q.backward()
    assert False, "no error"
except NotImplementedError as e:
    g.message(f"second derivative through linear_solve in a checkpoint raises: {e}")
del Q

#####################################################################
# smoothers with checkpoint=k in a frozen-shift force and in a
# prescribed-staple log-det force (family (i) of note XXV)
#####################################################################
even, odd = g.even_odd_projectors(grid)
a_gauge = g.qcd.gauge.action.iwasaki(5.5)
forces = []
for k in [None, 1, 2]:
    T = g.qcd.gauge.smear.directional_frozen_shift(
        U, 1, odd, fg.smoothed_staple_generator(1, odd, fg.transverse_stout(1, 0.1, 4, checkpoint=k), rho=0.2)
    )
    forces.append(a_gauge.transformed(T).gradient(U, U))
for k, F in zip([1, 2], forces[1:]):
    eps = rel(F, forces[0])
    g.message(f"frozen-shift force with checkpoint={k} vs without: {eps}")
    assert eps < 1e-14

mu = 2
dpt = g.qcd.gauge.smear.local_stout(rho=0.12, dimension=mu, checkerboard=g.even).transport(U)[0]
transporters = fg.frozen_transporters(mu, even, fg.staple_sum_paths(mu, 4))
ones = g.complex(grid)
ones[:] = 1
P0 = g(ones - even)


def staple(smoother):
    def C(x):
        F = [g(f * P0) if nu == mu else f for nu, f in enumerate(x[0:4])]
        T = transporters(smoother(F))
        s = T[0]
        for t in T[1:]:
            s = s + t
        return g(0.12 * s)

    return C


ld_forces = []
for k in [None, 1]:
    C = staple(fg.transverse_stout(mu, 0.1, 3, checkpoint=k))
    ld_forces.append(dpt.weighted_log_det_jacobian_gradient(U, U, ones, staple=C))
eps = rel(ld_forces[1], ld_forces[0])
g.message(f"prescribed-staple log-det force with checkpoint=1 vs without: {eps}")
assert eps < 1e-14

#####################################################################
# memory: the local passes of a chain of checkpoints keep no fields
# alive beyond their own pass (regression: the materialized X^dag of the
# exp towers of every local pass lived until the outermost pass ended, one
# field per exp and step).  A probe between the checkpoints counts, when
# the backward reaches it, the exp towers that still hold an X^dag; the
# count must not grow with the number of steps back-propagated.
#####################################################################
import gc
from gpt.ad.reverse.foundation.matrix.exp import _tower

held = []


def count_held():
    held.append(sum(1 for o in gc.get_objects() if isinstance(o, _tower) and o.xdag is not None))


probe = rad.primitive(
    "probe", lambda x: g.copy(x), lambda c: c, vjp=lambda i, flow, X: (count_held(), flow)[1], reads=((),)
)
n_steps = 8
aU = [rad.node(g.copy(u)) for u in U]
F = aU
for _ in range(n_steps):
    F = rad.checkpoint(fg.transverse_stout(2, 0.1, 1), F)
    F = [probe(F[0])] + F[1:]
action(F).backward()
g.message(f"exp towers holding X^dag during the backward of {n_steps} checkpoints: {held}")
assert len(held) == n_steps and max(held) <= held[0]
