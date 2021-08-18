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
import gpt as g
from gpt.core.group import differentiable_functional


def staple(U, mu):
    st = g.lattice(U[0])
    st[:] = 0
    Nd = len(U)
    for nu in range(Nd):
        if mu != nu:
            st += g.qcd.gauge.staple(U, mu, nu)
    return g(g.adj(st))


class wilson(differentiable_functional):
    def __init__(self, beta):
        self.beta = beta
        self.__name__ = f"wilson({beta})"

    def __call__(self, U):
        # Let beta = 2 ndim_repr / g^2
        #
        # S(U) = -beta sum_{mu>nu} Re[Tr[P_{mu,nu}]]/ndim_repr        (only U-dependent part)
        #      = -2/g^2 sum_{mu>nu} Re[Tr[P_{mu,nu}]]
        #      = -1/g^2 sum_{mu,nu} Re[Tr[P_{mu,nu}]]
        #      = -2/g^2 sum_{mu,nu} Re[Tr[staple_{mu,nu}^dag U_mu]]
        #
        # since   P_{mu,nu} = staple_{mu,nu}^dag U_mu + staple_{mu,nu} U_mu^dag = 2 Re[staple^dag * U]
        Nd = len(U)
        vol = U[0].grid.gsites
        return self.beta * (1.0 - g.qcd.gauge.plaquette(U)) * (Nd - 1) * Nd * vol / 2.0

    def gradient(self, U, dU):
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
        dS = []
        for Umu in dU:
            mu = U.index(Umu)
            dSdU_mu = staple(U, mu)
            dSdU_mu @= g.qcd.gauge.project.traceless_anti_hermitian(
                g(Umu * dSdU_mu)
            ) * (self.beta / (2.0 * Umu.otype.shape[0]) / 1j)
            dSdU_mu.otype = Umu.otype.cartesian()
            dS.append(dSdU_mu)
        return dS
