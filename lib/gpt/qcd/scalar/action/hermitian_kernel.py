#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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

#
# The idea of a complement kernel is to take any input kernel T
# and project it to sub-domains 0 and 1 via P0 and P1 (P0^2=P0, P1^2=P1, P0 + P1 = 1).
# After this projection, the kernel has Jacobi determinant one.
#
#   v' = (1 + P1 T P0) v = M v
#
#   M  = (1          0 )
#        ( T_01      1 )   -> det(M) = 1
#
# We then make it hermitian by considering a hermitian input kernel T and
#
#   X  = M^dag M = (1 + P0 T P1) (1 + P1 T P0)
#      = (1 + P0 T P1 + P1 T P0 + P0 T P1 T P0)
#
# with det(X) = 1.
#
# The corresponding kernel mass term is then given by
#
#    action = (1/2) P^dag X^2 P
#
# The role of X^2 is therefore that of the inverse mass term M^{-1} .
#


class complement:
    def __init__(self, U, f, P0, P1):
        self.f = f
        self.P0 = P0
        self.P1 = P1
        self.U = U
        self.nd = len(U)

    def __call__(self, dst, src):
        i0 = len(dst) - len(self.P0)

        src0 = src[:i0] + [g(self.P0[i] * src[i0 + i]) for i in range(len(self.P0))]
        self.f(dst, src0)
        for i in range(i0, len(src)):
            dst[i] *= self.P1[i - i0]
            dst[i] += src[i]

        src = g.copy(dst)
        src0 = src[:i0] + [g(self.P1[i] * src[i0 + i]) for i in range(len(self.P1))]
        self.f(dst, src0)
        for i in range(i0, len(src)):
            dst[i] *= self.P0[i - i0]
            dst[i] += src[i]

    def inverse(self, inverter):
        return g.qcd.gauge.algebra_laplace.inverse(self, inverter)

    def projected_gradient(self, left, U, right):
        assert len(right) == len(self.P0)
        right0 = [g(self.P0[i] * right[i]) for i in range(len(self.P0))]
        right1 = [g(self.P1[i] * right[i]) for i in range(len(self.P1))]
        left0 = [g(self.P0[i] * left[i]) for i in range(len(self.P0))]
        left1 = [g(self.P1[i] * left[i]) for i in range(len(self.P1))]
        grad10 = self.f.projected_gradient(left1, U, right0)
        grad01 = self.f.projected_gradient(left0, U, right1)

        T_right0 = g.copy(right0)
        self.f(g.copy(U) + T_right0, U + right0)
        for x in T_right0:
            x *= self.P1
        T_left0 = g.copy(left0)
        self.f(g.copy(U) + T_left0, U + left0)
        for x in T_left0:
            x *= self.P1

        grad_T2_a = self.f.projected_gradient(left0, U, T_right0)
        grad_T2_b = self.f.projected_gradient(T_left0, U, right0)
        return [g(grad10[i] + grad01[i] + grad_T2_a[i] + grad_T2_b[i]) for i in range(len(grad10))]


def mass_term(sqrt_inv_M, inverter):
    f_U = g.matrix_operator(
        mat=sqrt_inv_M,
        inv_mat=sqrt_inv_M.inverse(inverter),
        accept_list=True,
        accept_guess=(False, True),
    )

    f_U_sqr = f_U * f_U

    def f_sqr_projected_gradient(U, vec):
        # d(M^2) = M dM + dM M
        M_vec = g(f_U * (U + vec))[-len(vec) :]
        grad1 = sqrt_inv_M.projected_gradient(M_vec, U, vec)
        return [g(2 * x) for x in grad1]

    return g.qcd.scalar.action.general_mass_term(
        inv_M=f_U_sqr, sqrt_inv_M=f_U, inv_M_projected_gradient=f_sqr_projected_gradient
    )
