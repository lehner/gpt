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
from time import time


class fgcr:
    def __init__(self, params):
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.restartlen = params["restartlen"]

    def __call__(self, mat, src, psi, prec=None):
        # verbosity
        verbose = g.default.is_verbose("fgcr")
        checkres = True  # for now

        # total time
        tt0 = time()

        # parameters
        rlen = self.restartlen

        # tensors
        dtype_r, dtype_c = np.float64, np.complex128
        alpha = np.empty((rlen), dtype_c)
        beta = np.empty((rlen, rlen), dtype_c)
        gamma = np.empty((rlen), dtype_r)
        delta = np.empty((rlen), dtype_c)

        # fields
        r, mmr, mmpsi = g.copy(src), g.copy(src), g.copy(src)
        p = [g.lattice(src) for i in range(rlen)]
        mmp = [g.lattice(src) for i in range(rlen)]

        # residual target
        ssq = g.norm2(src)
        rsq = self.eps**2. * ssq

        # initial values
        r2 = self.restart(mat, psi, mmpsi, src, r)

        for k in range(self.maxiter):
            # iteration within current krylov space
            i = k % rlen

            # iteration criteria
            reached_maxiter = k+1 == self.maxiter
            need_restart = i+1 == rlen

            t0 = time()
            if not prec is None:
                prec(r, p[i])
            else:
                p[i] @= r
            t1 = time()

            t2 = time()
            mat(p[i], mmp[i])
            t3 = time()

            t4 = time()
            g.orthogonalize(mmp[i], mmp[0:i], beta[:, i])
            t5 = time()

            t6 = time()
            ip, mmp2 = g.innerProductNorm2(mmp[i], r)
            gamma[i] = mmp2**0.5

            if gamma[i] == 0.:
                g.message("fgcr breakdown, gamma[%d] = 0" % (i))
                break

            mmp[i] /= gamma[i]
            alpha[i] = ip / gamma[i]
            r2 = g.axpy_norm2(r, -alpha[i], mmp[i], r)
            t7 = time()

            if verbose:
                g.message(
                    "Timing[s]: Prec = %g, Matrix = %g, Orthog = %g, Rest = %g"
                    % (t1 - t0, t3 - t2, t5 - t4, t7 - t6))
                g.message("res^2[ %d, %d ] = %g" % (k, i, r2))

            if r2 <= rsq or need_restart or reached_maxiter:
                self.update_psi(psi, alpha, beta, gamma, delta, p, i)

                if r2 <= rsq:
                    if verbose:
                        tt1 = time()
                        g.message("Converged in %g s" % (tt1 - tt0))
                        if checkres:
                            res = self.calc_res(mat, psi, mmpsi, src, r) / ssq
                            g.message(
                                "Computed res = %g, true res = %g, target = %g"
                                % (r2**0.5, res**0.5, self.eps))
                    break

                if reached_maxiter:
                    if verbose:
                        tt1 = time()
                        g.message("Did NOT converge in %g s" % (tt1 - tt0))
                        if checkres:
                            res = self.calc_res(mat, psi, mmpsi, src, r) / ssq
                            g.message(
                                "Computed res = %g, true res = %g, target = %g"
                                % (r2**0.5, res**0.5, self.eps))

                if need_restart:
                    r2 = self.restart(mat, psi, mmpsi, src, r)
                    if verbose:
                        g.message("Performed restart")

    def update_psi(self, psi, alpha, beta, gamma, delta, p, i):
        # backward substitution
        for j in reversed(range(i + 1)):
            delta[j] = (alpha[j] - np.dot(beta[j, j+1:i+1], delta[j+1:i+1])) / gamma[j]

        for j in range(i + 1):
            psi += delta[j] * p[j]

    def restart(self, mat, psi, mmpsi, src, r):
        return self.calc_res(mat, psi, mmpsi, src, r)

    def calc_res(self, mat, psi, mmpsi, src, r):
        mat(psi, mmpsi)
        return g.axpy_norm2(r, -1., mmpsi, src)
