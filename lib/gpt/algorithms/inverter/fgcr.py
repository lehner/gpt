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


class fgcr(base_iterative):
    @g.params_convention(eps=1e-15, maxiter=1000000, restartlen=20, checkres=True, prec=None)
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.restartlen = params["restartlen"]
        self.checkres = params["checkres"]
        self.prec = params["prec"]

    def modified(self, **params):
        return fgcr({**self.params, **params})

    def update_psi(self, psi, alpha, beta, gamma, chi, p, i):
        # backward substitution
        for j in reversed(range(i + 1)):
            chi[j] = (alpha[j] - np.dot(beta[j, j + 1 : i + 1], chi[j + 1 : i + 1])) / gamma[j]

        for j in range(i + 1):
            psi += chi[j] * p[j]

    def restart(self, mat, psi, mmpsi, src, r, p):
        if self.prec is not None:
            for v in p:
                v[:] = 0
        return self.calc_res(mat, psi, mmpsi, src, r)

    def calc_res(self, mat, psi, mmpsi, src, r):
        mat(mmpsi, psi)
        return g.axpy_norm2(r, -1.0, mmpsi, src)

    def __call__(self, mat):
        vector_space = None
        if isinstance(mat, g.matrix_operator):
            vector_space = mat.vector_space
            mat = mat.mat
            # remove wrapper for performance benefits

        prec = self.prec(mat) if self.prec is not None else None

        @self.timed_function
        def inv(psi, src, t):
            t("setup")

            # parameters
            rlen = self.restartlen

            # tensors
            dtype_r, dtype_c = g.double.real_dtype, g.double.complex_dtype
            alpha = np.empty((rlen), dtype_c)
            beta = np.empty((rlen, rlen), dtype_c)
            gamma = np.empty((rlen), dtype_r)
            chi = np.empty((rlen), dtype_c)

            # fields
            r, mmpsi = g.copy(src), g.copy(src)
            p = [g.lattice(src) for i in range(rlen)]
            z = [g.lattice(src) for i in range(rlen)]

            # initial residual
            r2 = self.restart(mat, psi, mmpsi, src, r, p)

            # source
            ssq = g.norm2(src)
            if ssq == 0.0:
                assert r2 != 0.0  # need either source or psi to not be zero
                ssq = r2

            # target residual
            rsq = self.eps**2.0 * ssq

            for k in range(self.maxiter):
                # iteration within current krylov space
                i = k % rlen

                # iteration criteria
                need_restart = i + 1 == rlen

                t("prec")
                if prec is not None:
                    p[i][:] = 0
                    prec(p[i], r)
                else:
                    p[i] @= r

                t("mat")
                mat(z[i], p[i])

                t("ortho")
                g.orthogonalize(z[i], z[0:i], beta[:, i])

                t("linalg")
                ip, z2 = g.inner_product_norm2(z[i], r)
                gamma[i] = z2**0.5
                if gamma[i] == 0.0:
                    self.debug(f"breakdown, gamma[{i:d}] = 0")
                    break
                z[i] /= gamma[i]
                alpha[i] = ip / gamma[i]
                r2 = g.axpy_norm2(r, -alpha[i], z[i], r)

                t("other")
                self.log_convergence((k, i), r2, rsq)

                if r2 <= rsq or need_restart:
                    t("update_psi")
                    self.update_psi(psi, alpha, beta, gamma, chi, p, i)

                if r2 <= rsq:
                    msg = f"converged in {k+1} iterations;  computed squared residual {r2:e} / {rsq:e}"
                    if self.checkres:
                        res = self.calc_res(mat, psi, mmpsi, src, r)
                        msg += f";  true squared residual {res:e} / {rsq:e}"
                    self.log(msg)
                    return

                if need_restart:
                    t("restart")
                    r2 = self.restart(mat, psi, mmpsi, src, r, p)
                    self.debug("performed restart")

            msg = f"NOT converged in {k+1} iterations;  computed squared residual {r2:e} / {rsq:e}"
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
