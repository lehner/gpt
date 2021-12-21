#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2021  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


class base(differentiable_functional):

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
        # define staple here as adjoint
        dS = []
        for Umu in dU:
            mu = U.index(Umu)
            dSdU_mu = self.staple(U, mu)
            dSdU_mu @= g.qcd.gauge.project.traceless_anti_hermitian(
                g(Umu * g.adj(dSdU_mu))
            ) * (1.0 / 2.0 / 1j)
            dSdU_mu.otype = Umu.otype.cartesian()
            dS.append(dSdU_mu)
        return dS
