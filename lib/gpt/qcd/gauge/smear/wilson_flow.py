#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import numpy as np
import gpt as g
from gpt.params import params_convention


def staple(U, mu):
    st = g.lattice(U[0])
    st[:] = 0
    Nd = len(U)
    for nu in range(Nd):
        if mu != nu:
            st += g.qcd.gauge.staple(U, mu, nu)
    return g(g.adj(st))


def dSdU(U):
    # S(U) = -beta/Nc sum_{mu>nu} Re[Tr[P_{mu,nu}]]
    #      = -2/g^2 sum_{mu>nu} Re[Tr[P_{mu,nu}]]
    #      = -1/g^2 sum_{mu,nu} Re[Tr[P_{mu,nu}]]
    #      = -2/g^2 sum_{mu,nu} Re[Tr[staple_{mu,nu}^dag U_mu]]
    #
    # since   P_{mu,nu} = staple_{mu,nu}^dag U_mu + staple_{mu,nu} U_mu^dag = 2 Re[staple^dag * U]
    #
    # Eq. (1.3) and Appendix A of https://link.springer.com/content/pdf/10.1007/JHEP08(2010)071.pdf
    # S(Umu) = -2/g^2 Re trace(Umu * staple)
    # dS(Umu) = lim_{eps->0} Ta ( S(e^{eps Ta} Umu) - S(Umu) ) / eps  with  \Tr[T_a T_b]=-1/2 \delta_{ab}
    # dS(Umu) = -2/g^2 T_a Re trace(T_a * Umu * staple)
    #         = -2/g^2 T_a 1/2 trace(T_a * Umu * staple + adj(staple) * adj(Umu) * adj(Ta))
    #         = -2/g^2 T_a 1/2 trace(T_a * (Umu * staple - adj(staple) * adj(Umu)))
    #         = -2/g^2 T_a 1/2 trace(T_a * (Umu * staple - adj(Umu*staple)))
    #         = -2/g^2 T_a trace(T_a * r0)    with r0 = 1/2(Umu * staple - adj(Umu*staple))
    # r0 = c_a T_a + imaginary_diagonal   with A^dag = -A
    # trace(T_a * r0) = -1/2 c_a
    # dS(Umu) = 1/g^2 tracelss_anti_hermitian(Umu * staple)
    dSdU = []
    for mu, Umu in enumerate(U):
        dSdU_mu = staple(U, mu)
        dSdU_mu @= g.qcd.gauge.project.traceless_anti_hermitian(g(Umu * dSdU_mu)) * (
            -1.0
        )
        dSdU.append(dSdU_mu)
    return dSdU  # deriv = -g^2 dSdU


def update_field(P, U, ep):
    for Pmu, Umu in zip(P, U):
        Umu @= g.project(g.matrix.exp(g(ep * Pmu)), "defect") * Umu


def add_field(Z, dU):
    for Zmu, dUmu in zip(Z, dU):
        Zmu += dUmu


def wilson_flow(U, epsilon):
    # Flow a gauge field from U(t) to U(t+epsilon) using the original Runke-Kutta scheme
    # Eq. (C.2) of https://link.springer.com/content/pdf/10.1007/JHEP08(2010)071.pdf
    Uprime = g.copy(U)
    Z = dSdU(Uprime)
    for Zmu in Z:
        Zmu *= 0.25
    update_field(Z, Uprime, epsilon)
    # Z = Z0/4
    # U = W1
    for Zmu in Z:
        Zmu *= -17.0 / 8.0
    # Z = -Z0*17/32
    add_field(Z, dSdU(Uprime))
    # Z = -Z0*17/32 + Z1
    for Zmu in Z:
        Zmu *= 8.0 / 9.0
    # Z = -Z0*17/36 + 8/9*Z1
    update_field(Z, Uprime, epsilon)
    # U = W2
    for Zmu in Z:
        Zmu *= -4.0 / 3.0
    # Z = Z0*17/27 - 32/27*Z1
    add_field(Z, dSdU(Uprime))
    # Z = Z0*17/27 - 32/27*Z1 + Z2
    for Zmu in Z:
        Zmu *= 3.0 / 4.0
    # Z = Z0*17/36 - 8/9*Z1 + 3/4*Z2
    update_field(Z, Uprime, epsilon)
    return Uprime
