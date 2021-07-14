#!/usr/bin/env python3
#
# Authors: Christoph Lehner, Mattia Bruno 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np

grid = g.grid([8,8,8,8], g.double)
rng = g.random("deriv")

U = g.qcd.gauge.unit(grid)
rng.normal_element(U)

p = {
    "kappa": 0.137,
    "csw_r": 0.0,
    "csw_t": 0.0,
    "xi_0": 1,
    "nu": 1,
    "isAnisotropic": False,
    "boundary_phases": [1., 1., 1., 1.],
}
Dops = [g.qcd.fermion.wilson_clover(U, p)]
mobius_params = {
    "mass": 0.08,
    "M5": 1.8,
    "b": 1.5,
    "c": 0.5,
    "Ls": 12,
    "boundary_phases": [1.0, 1.0, 1.0, 1.0],
}
Dops += [g.qcd.fermion.mobius(U, mobius_params)]

mom = g.group.cartesian(U)
rng.normal_element(mom)

#### numerical derivative
Uprime = g.qcd.gauge.unit(grid)
eps = 1e-2

# F = psi^dag Mdag M psi = (chi, chi) ; chi = M psi
# dF = psi^dag dMdag M psi + psi^dag Mdag dM psi
#    = (psi, dMdag chi) + (chi, dM psi)
def assert_gradient(mat, method, grid, dmat, dmatdag):
    psi = g.vspincolor(grid)
    chi = g.vspincolor(grid)
    rng.normal(psi)

    getattr(mat, method)(chi, psi)
    frc = dmat(chi, psi)
    tmp = dmatdag(psi, chi)
    
    dS_ex = 0.0
    for mu in range(len(frc)):
        frc[mu] @= frc[mu] + tmp[mu]
        dS_ex += g.group.inner_product(frc[mu], mom[mu])
    del frc, tmp
    
    dS_num = 0.0
    for coeff in [(1.0, 2./3.), (2.0, -1./12.), (-1.0,-2./3.), (-2.0,1/12)]:
    #for coeff in [(1.0, 0.5), (-1.0, -0.5)]:
        f = eps * coeff[0]
        for mu in range(len(Uprime)):
            Uprime[mu] @= g.group.compose(g(f * mom[mu]), U[mu])

        Mprime = mat.updated(Uprime)
        #chi @= Mprime * psi
        getattr(Mprime, method)(chi, psi)
        dS_num += (coeff[1]/eps) * g.norm2(chi)

    g.message(f"Exact gradient {dS_ex:.6e}")
    g.message(f"Numer gradient {dS_num:.6e}")
    g.message(f"Difference {abs(dS_num - dS_ex)/abs(dS_ex):.6e}")

    assert abs(dS_num - dS_ex)/abs(dS_ex) < 1e-7
    
for M in Dops:
    g.message(M.name, ' full M')
    assert_gradient(M, 'mat', M.F_grid, M.Mderiv, M.MderivDag)      
