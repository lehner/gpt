#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2024  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                        Adopted from Grid's BlockConjugateGradient
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
import numpy as np
from gpt.algorithms import base_iterative


def QC_is_R(Q, R):
    # checked QC_is_R(Q, R) = C  such that QC = R
    m_rr = g.inner_product(R, R)
    m_rr = 0.5 * (m_rr + m_rr.transpose().conjugate())
    L = np.linalg.cholesky(m_rr)
    C = L.transpose().conjugate()
    Cinv = np.linalg.inv(C)
    g.linear_combination(Q, R, np.ascontiguousarray(Cinv.transpose()))
    return C


class block_cg(base_iterative):
    @g.params_convention(eps=1e-15, maxiter=1000000, eps_abs=None, miniter=0)
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.eps_abs = params["eps_abs"]
        self.maxiter = params["maxiter"]
        self.miniter = params["miniter"]

    def modified(self, **params):
        return block_cg({**self.params, **params})

    def __call__(self, mat):
        vector_space = None
        if isinstance(mat, g.matrix_operator):
            vector_space = mat.vector_space
            mat = mat.specialized_list_callable()

        @self.timed_function
        def inv(X, B, t):
            nblock = len(B)

            t("reductions")
            ssq = g.norm2(B)

            AD = [g.lattice(x) for x in B]
            Q = [g.lattice(x) for x in B]
            Z = [g.lattice(x) for x in B]

            # QC = R = B-AX, D = Q
            t("mat")
            mat(AD, X)

            t("linear")
            tmp = g(g.expr(B) - g.expr(AD))

            t("QR")
            m_C = QC_is_R(Q, tmp)
            D = g.copy(Q)

            for k in range(self.maxiter):
                # Z  = AD
                t("mat")
                mat(Z, D)

                # M  = [D^dag Z]^{-1}
                t("reduction")
                m_DZ = g.inner_product(D, Z)

                t("inverse")
                m_M = np.linalg.inv(m_DZ)

                # X += D MC
                t("linear")
                m_tmp = m_M @ m_C
                g.linear_combination(tmp, D, np.ascontiguousarray(m_tmp.transpose()))
                for i in range(nblock):
                    X[i] += tmp[i]

                # QS = Q - ZM = Q - tmp
                g.linear_combination(tmp, Z, np.ascontiguousarray(m_M.transpose()))
                for i in range(nblock):
                    tmp[i] @= Q[i] - tmp[i]

                t("QR")
                m_S = QC_is_R(Q, tmp)

                # D = Q + D S^dag
                t("linear")
                m_tmp = m_S.transpose().conjugate()
                g.linear_combination(tmp, D, np.ascontiguousarray(m_tmp.transpose()))
                for i in range(nblock):
                    D[i] @= Q[i] + tmp[i]

                # C = S C
                m_C = m_S @ m_C

                m_rr = m_C.transpose().conjugate() @ m_C

                max_resid_rel = 0
                max_resid_abs = 0
                for b in range(nblock):
                    rr = m_rr[b, b].real
                    if rr > max_resid_abs:
                        max_resid_abs = rr
                    rr /= ssq[b]
                    if rr > max_resid_rel:
                        max_resid_rel = rr

                self.log_convergence(k, max_resid_rel, self.eps**2.0)
                if k + 1 >= self.miniter:
                    if self.eps_abs is not None and max_resid_abs <= self.eps_abs**2.0:
                        self.log(f"converged in {k+1} iterations (absolute criterion)")
                        return
                    if max_resid_rel <= self.eps**2.0:
                        self.log(f"converged in {k+1} iterations")
                        return

            self.log(
                f"NOT converged in {k+1} iterations;  squared resudial relative {max_resid_rel} and absolute {max_resid_abs}"
            )

        return g.matrix_operator(
            mat=inv,
            inv_mat=mat,
            accept_guess=(True, False),
            vector_space=vector_space,
            accept_list=True,
        )
