#!/usr/bin/env python3
#
# Authors: Christoph Lehner
#
import gpt as g
import sys, os
import numpy as np

rad = g.ad.reverse

beta = 2.95

U0 = g.load("cdrhmc_16_0.5625x3/ckpoint_lat.99")
even, odd = g.even_odd_projectors(U0[0].grid)
full = g(even + odd)
none = g(0 * full)


def apply_smearing(U, rho_val, inverse):

    description = [
        [(rho_val, g.path().f(nu).f(mu).b(nu, 4).b(mu).f(nu, 3)) for nu in range(4) if mu != nu]
        for mu in range(4)
    ]

    pt_e = [
        g.qcd.gauge.smear.parallel_transport(
            U,
            description,
            [even if i == j else full for i in range(4)],
            [odd if i == j else none for i in range(4)],
        )
        for j in range(4)
    ]

    pt_o = [
        g.qcd.gauge.smear.parallel_transport(
            U,
            description,
            [odd if i == j else full for i in range(4)],
            [even if i == j else none for i in range(4)],
        )
        for j in range(4)
    ]

    inv_pt_e = [x.inv() for x in pt_e]
    inv_pt_o = [x.inv() for x in pt_o]

    # undo smearing
    if inverse:
        for i in reversed(range(4)):
            U = inv_pt_e[i](U)
            U = inv_pt_o[i](U)
    else:
        for i in range(4):
            U = pt_o[i](U)
            U = pt_e[i](U)

    return U


# then learn
rho_val = -0.3 + 0j
for epoch in range(20):

    # unsmear
    _U = apply_smearing(U0, rho_val, True)

    # first create cost function graph with integration variable as leaf
    # mark inner node as infinitesimal instead of cartesian (dU versus dU U^dagger)
    # since we want to re-use its gradient as an initial value to propagate through
    # the initial inverse smearing
    nnU0 = [rad.node(rad.node(u, infinitesimal_to_cartesian=False)) for u in _U]
    rho = rad.node(rho_val)
    g.qcd.gauge.action.differentiable_iwasaki(beta)(apply_smearing(nnU0, rad.node(rho), False))()
    c = sum(g.norm2(nnU0[mu].gradient) for mu in range(4)) / 4 / full.grid.fsites / 8 / 3

    # then create a graph from physical field to integration variable as leaf
    nU = [rad.node(u) for u in U0]
    rho2 = rad.node(rho_val)
    nU0 = apply_smearing(nU, rho2, True)

    cv = c()
    rho_gradients = [rho.gradient]

    # there is also a gradient on the physical fields that needs to be propagated to the rho in the inverse smearing
    for mu in range(4):
        nU0[mu](initial_gradient=nnU0[mu].value.gradient)
        rho_gradients.append(rho2.gradient)

    g.message(
        f"""

    Epoch {epoch} has:

      Cost function   {cv}
      Rho             {rho_val}
      Rho gradients   {rho_gradients}   ->   sum to  {sum(rho_gradients)}

    """
    )

    rho_val -= 2e-2 * sum(rho_gradients)


#
# V = f^{-1}(U, rho)
# S(f(V, rho)),    c = | \partial S(f(V, rho)) / \partial V |^2
#
# dc/drho
#
