#!/usr/bin/env python3
import gpt as g
import numpy as np

rad = g.ad.reverse

U = g.load("cdrhmc_16_0.5625x3/ckpoint_lat.99")

use_unit = False

if use_unit:
    U = g.qcd.gauge.unit(U[0].grid)

action = g.qcd.gauge.action.differentiable_iwasaki(2.95)

nnU = [rad.node(rad.node(u)) for u in U]

nA = [rad.node(g.group.cartesian(u)) for u in U]

# First create compute graph and \partial S / \partial U, stored in nnU[mu].gradient
action(nnU)()


def Hessian_vec(src):
    # then create expression for inner product with right-hand side
    for mu in range(4):
        nA[mu].value @= src[mu]
    c = sum(g.inner_product(nnU[mu].gradient, nA[mu]) for mu in range(4))
    # and do forward and backward propagation
    c()
    # this now is \partial <\partial S / \partial U, src> / \partial U,
    # i.e., the Hessian applied to the vector src
    return [nnU[mu].value.gradient for mu in range(4)]


# create operator that stacks Lorentz index in 4-th dimension
cache = {}


def Hessian_vec_5d(dst5d, src5d):
    src = g.separate(src5d, 4, cache)
    dst = Hessian_vec(src)
    dst5d @= g.merge(dst)


H = g.matrix_operator(mat=Hessian_vec_5d)

test_src = g.merge(g.group.cartesian(U))
g.random("test").element(test_src)

# now run Lanczos
irl = g.algorithms.eigen.irl(
    Nk=5,
    Nstop=5,
    Nm=30,
    resid=1e-8,
    betastp=1e-5,
    maxiter=30,
    Nminres=0,
    sort_eigenvalues=lambda x: sorted(x),
)

if False:
    # upper edge of spectrum
    evec, evals = irl(H, test_src)
    g.message(evals)
    evec_max_norm2 = g(g.trace(g.adj(evec[0]) * evec[0]))
    if not use_unit:
        np.savetxt("H_eval_max", evals)
        np.savetxt("H_evec_max_2d", evec_max_norm2[:, :, 0, 0, 0].real.reshape(8, 8))

if False:
    eval_max = 11.27393345529601
    if use_unit:
        eval_max = 15
    H_inv = g.matrix_operator(mat=lambda dst, src: g.eval(dst, eval_max * src - H * src))
    # upper edge of spectrum
    evec, evals = irl(H_inv, test_src)
    g.message(evals)
    evec_min_norm2 = g(g.trace(g.adj(evec[0]) * evec[0]))
    if not use_unit:
        np.savetxt("H_eval_min", eval_max - np.array(evals))
        np.savetxt("H_evec_min_2d", evec_min_norm2[:, :, 0, 0, 0].real.reshape(8, 8))

if True:
    # test Hessian
    dA = g.random("test").normal_element(g.group.cartesian(U))
    eps = 1e-6

    a0 = action(U)
    a1 = action([g(g.group.compose(g(eps * dA[mu]), U[mu])) for mu in range(4)])
    F = g.qcd.gauge.action.iwasaki(2.95).gradient(U, U)
    F_dA = sum(g.group.inner_product(F[mu], dA[mu]) for mu in range(4))
    H_dA = Hessian_vec(dA)
    dA_H_dA = sum(g.group.inner_product(dA[mu], H_dA[mu]) for mu in range(4))
    print((a1 - a0) / eps)
    print((a1 - a0 - eps * F_dA) / eps)
    print((a1 - a0 - eps * F_dA - eps**2 * dA_H_dA) / eps)

    # NOTE: test makes clear that H is defined only up to normalization factor 2
    # of group generators that differs between the proper g.group.inner_product
    # and the currently differentiable g.inner_product
