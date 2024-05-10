#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann (daniel.richtmann@ur.de)
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


class cagcr(base_iterative):
    """
    This algorithm delays the orthogonalization by 'unrolling' the iterations within a restart of standard gcr.
    Due to this delay the restart length cannot be chosen arbitrarily large as the vectors quickly become linear dependent otherwise.
    The convergence greatly varies depending on this number but something in the ballpark of 8-10 should be be a good value here.
    This is acceptible since this algorithm isn't aimed to be used as a standalone solver but rather as a preconditioner to a flexible solver or a smoother/coarse solver in multigrid.
    """

    @g.params_convention(eps=1e-15, maxiter=1000000, restartlen=10, checkres=True)
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.restartlen = params["restartlen"]
        self.checkres = params["checkres"]

    def modified(self, **params):
        return cagcr({**self.params, **params})

    def calc_res(self, mat, psi, mmpsi, src, r):
        mat(mmpsi, psi)
        return g.axpy_norm2(r, -1.0, mmpsi, src)

    def __call__(self, mat):
        vector_space = None
        if isinstance(mat, g.matrix_operator):
            vector_space = mat.vector_space
            mat = mat.mat
            # remove wrapper for performance benefits

        @self.timed_function
        def inv(psi, src, t):
            t("setup")

            # parameters
            rlen = self.restartlen

            # tensors
            alpha = np.empty((rlen), g.double.complex_dtype)

            # fields
            r, mmpsi = g.copy(src), g.copy(src)
            p = [g.lattice(src) for i in range(rlen + 1)]
            # in QUDA, q is just an "alias" to p with q[k] = p[k+1]
            # don't alias here, but just use slicing

            # initial residual
            r2 = self.calc_res(mat, psi, mmpsi, src, r)
            p[0] @= r

            # source
            ssq = g.norm2(src)
            if ssq == 0.0:
                assert r2 != 0.0  # need either source or psi to not be zero
                ssq = r2

            # target residual
            rsq = self.eps**2.0 * ssq

            for k in range(0, self.maxiter, rlen):
                t("mat")
                for i in range(rlen):
                    mat(p[i + 1], p[i])

                t("inner_product")
                ips = g.inner_product(p[1:], p[1:] + [p[0]])  # single reduction

                t("solve")
                rhs = ips[:, -1]  # last column
                A = ips[:, :-1]  # all but last column
                alpha = np.linalg.solve(A, rhs)

                # # check that solution is correct
                # g.message(np.allclose(np.dot(A, alpha), rhs))

                t("update_psi")
                for i in range(rlen):
                    g.axpy(psi, alpha[i], p[i], psi)

                if self.maxiter != rlen:
                    t("update_residual")
                    for i in range(rlen):
                        g.axpy(r, -alpha[i], p[i + 1], r)

                    t("residual")
                    r2 = g.norm2(r)

                    t("other")
                    self.log_convergence(k, r2, rsq)

                if r2 <= rsq:
                    msg = f"converged in {k+rlen} iterations"
                    if self.maxiter != rlen:
                        msg += f";  computed squared residual {r2:e} / {rsq:e}"
                    if self.checkres:
                        res = self.calc_res(mat, psi, mmpsi, src, r)
                        msg += f";  true squared residual {res:e} / {rsq:e}"
                    self.log(msg)
                    return

                if self.maxiter != rlen:
                    t("restart")
                    p[0] @= r
                    self.debug("performed restart")

            msg = f"NOT converged in {k+rlen} iterations"
            if self.maxiter != rlen:
                msg += f";  computed squared residual {r2:e} / {rsq:e}"
            if self.checkres:
                res = self.calc_res(mat, psi, mmpsi, src, r)
                msg += f";  true squared residual {res:e} / {rsq:e}"
            self.log(msg)

        return g.matrix_operator(
            mat=inv,
            inv_mat=mat,
            vector_space=vector_space,
            accept_guess=(True, False),
        )
