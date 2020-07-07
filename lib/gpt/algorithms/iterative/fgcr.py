#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann
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


class fgcr:
    @g.params_convention(eps=1e-15, maxiter=1000000)
    def __init__(self, params):
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.restartlen = params["restartlen"]
        self.prec = params["prec"] if "prec" in params else None
        self.history = None

    def update_psi(self, psi, alpha, beta, gamma, delta, p, i):
        # backward substitution
        for j in reversed(range(i + 1)):
            delta[j] = (
                alpha[j] - np.dot(beta[j, j + 1 : i + 1], delta[j + 1 : i + 1])
            ) / gamma[j]

        for j in range(i + 1):
            psi += delta[j] * p[j]

    def restart(self, mat, psi, mmpsi, src, r):
        return self.calc_res(mat, psi, mmpsi, src, r)

    def calc_res(self, mat, psi, mmpsi, src, r):
        mat(mmpsi, psi)
        return g.axpy_norm2(r, -1.0, mmpsi, src)

    def __call__(self, mat):
        def inv(psi, src):
            self.history = []
            # verbosity
            verbose = g.default.is_verbose("fgcr")
            checkres = True  # for now

            # timing
            t = g.timer()

            # start clocks
            t.start("total")
            t.start("setup")

            # parameters
            rlen = self.restartlen

            # tensors
            dtype_r, dtype_c = np.float64, np.complex128
            alpha = np.empty((rlen), dtype_c)
            beta = np.empty((rlen, rlen), dtype_c)
            gamma = np.empty((rlen), dtype_r)
            delta = np.empty((rlen), dtype_c)

            # fields
            r, mmpsi = g.copy(src), g.copy(src)
            p = [g.lattice(src) for i in range(rlen)]
            mmp = [g.lattice(src) for i in range(rlen)]

            # residual target
            ssq = g.norm2(src)
            rsq = self.eps ** 2.0 * ssq

            # initial values
            r2 = self.restart(mat, psi, mmpsi, src, r)

            t.stop("setup")

            for k in range(self.maxiter):
                # iteration within current krylov space
                i = k % rlen

                # iteration criteria
                reached_maxiter = k + 1 == self.maxiter
                need_restart = i + 1 == rlen

                t.start("prec")
                if self.prec is not None:
                    self.prec(p[i], r)
                else:
                    p[i] @= r
                t.stop("prec")

                t.start("mat")
                mat(mmp[i], p[i])
                t.stop("mat")

                t.start("ortho")
                g.orthogonalize(mmp[i], mmp[0:i], beta[:, i])
                t.stop("ortho")

                t.start("linalg")
                ip, mmp2 = g.innerProductNorm2(mmp[i], r)
                gamma[i] = mmp2 ** 0.5

                if gamma[i] == 0.0:
                    g.message("fgcr breakdown, gamma[%d] = 0" % (i))
                    break

                mmp[i] /= gamma[i]
                alpha[i] = ip / gamma[i]
                r2 = g.axpy_norm2(r, -alpha[i], mmp[i], r)
                self.history.append(r2)
                t.stop("linalg")

                if verbose:
                    g.message("res^2[ %d, %d ] = %e" % (k, i, r2))

                if r2 <= rsq or need_restart or reached_maxiter:
                    t.start("update_psi")
                    self.update_psi(psi, alpha, beta, gamma, delta, p, i)
                    t.stop("update_psi")

                    if r2 <= rsq:
                        if verbose:
                            t.stop("total")
                            g.message("Converged in %g s" % (t.dt["total"]))
                            t.print("fgcr")
                            if checkres:
                                comp_res = r2 / ssq
                                res = self.calc_res(mat, psi, mmpsi, src, r) / ssq
                                g.message(
                                    "Computed res = %g, true res = %g, target = %g"
                                    % (comp_res ** 0.5, res ** 0.5, self.eps)
                                )
                        break

                    if reached_maxiter:
                        if verbose:
                            t.stop("total")
                            g.message("Did NOT converge in %g s" % (t.dt["total"]))
                            t.print("fgcr")
                            if checkres:
                                comp_res = r2 / ssq
                                res = self.calc_res(mat, psi, mmpsi, src, r) / ssq
                                g.message(
                                    "Computed res = %g, true res = %g, target = %g"
                                    % (comp_res ** 0.5, res ** 0.5, self.eps)
                                )

                    if need_restart:
                        t.start("restart")
                        r2 = self.restart(mat, psi, mmpsi, src, r)
                        if verbose:
                            g.message("Performed restart")
                        t.stop("restart")

        otype = None
        grid = None
        if type(mat) == g.matrix_operator:
            otype = mat.otype
            grid = mat.grid

        return g.matrix_operator(
            mat=inv, inv_mat=mat, otype=otype, zero=(False, False), grid=grid
        )
