#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2022  Raphael Lehner (raphael.lehner@physik.uni-regensburg.de)
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
from gpt.algorithms.eigen.arnoldi import arnoldi_iteration


class fom(base_iterative):
    @g.params_convention(eps=1e-15, maxiter=1000, restartlen=20, checkres=True)
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.restartlen = params["restartlen"]
        self.checkres = params["checkres"]

    def solve_hessenberg(self, H, r2):
        n = len(H)
        b = np.zeros(n, np.complex128)
        b[0] = r2**0.5
        for i in range(n - 1):
            k = -H[i][-1] / H[i][-2]
            for j in range(n - i):
                H[i + j][i + 1] += k * H[i + j][i]
            b[i + 1] += k * b[i]
        y = np.zeros(n, np.complex128)
        for i in reversed(range(n)):
            y[i] = b[i] / H[i][i]
            for j, hj in enumerate(H[i][0:-1]):
                b[j] -= hj * y[i]
        rn = -H[-1][-1] * y[-1]
        return y, rn

    def calc_res(self, mat, psi, mmp, src, r):
        mat(mmp, psi)
        return g.axpy_norm2(r, -1.0, mmp, src)

    def __call__(self, mat):
        vector_space = None
        if isinstance(mat, g.matrix_operator):
            vector_space = mat.vector_space
            mat = mat.mat

        @self.timed_function
        def inv(psi, src, t):
            assert src != psi

            t("setup")
            rlen = self.restartlen
            mmp, r = g.copy(src), g.copy(src)
            r2 = self.calc_res(mat, psi, mmp, src, r)

            ssq = g.norm2(src)
            if ssq == 0.0:
                assert r2 != 0.0
                ssq = r2
            rsq = self.eps**2.0 * ssq

            g.default.push_verbose("arnoldi", False)
            a = arnoldi_iteration(mat, r)
            g.default.pop_verbose()

            for k in range(0, self.maxiter, rlen):
                t("arnoldi")
                for i in range(rlen):
                    # for sufficiently small restart length
                    # should not need second orthogonalization
                    # step
                    a(second_orthogonalization=False)
                Q, H = a.basis, a.H

                t("solve_hessenberg")
                y, rn = self.solve_hessenberg(H, r2)

                t("update_psi")
                g.linear_combination(mmp, Q[0:-1], y)
                psi += mmp

                if self.maxiter != rlen:
                    t("update_res")
                    r @= g.eval(Q[-1] * rn)

                t("residual")
                r2 = np.abs(rn) ** 2.0

                t("other")
                self.log_convergence(k, r2, rsq)

                if r2 <= rsq:
                    msg = f"converged in {k+rlen} iterations"
                    if self.maxiter != rlen:
                        msg += f";  computed squared residual {r2:e} / {rsq:e}"
                    if self.checkres:
                        res = self.calc_res(mat, psi, mmp, src, r)
                        msg += f";  true squared residual {res:e} / {rsq:e}"
                    self.log(msg)
                    return

                if self.maxiter != rlen:
                    t("restart")
                    a.basis = [Q[-1]]
                    a.H = []
                    self.debug("performed restart")

            msg = f"NOT converged in {k+rlen} iterations"
            if self.maxiter != rlen:
                msg += f";  computed squared residual {r2:e} / {rsq:e}"
            if self.checkres:
                res = self.calc_res(mat, psi, mmp, src, r)
                msg += f";  true squared residual {res:e} / {rsq:e}"
            self.log(msg)

        return g.matrix_operator(
            mat=inv, inv_mat=mat, accept_guess=(True, False), vector_space=vector_space
        )
