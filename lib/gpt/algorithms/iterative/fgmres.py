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


class fgmres:
    def __init__(self, params):
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.restartlen = params["restartlen"]

    def __call__(self, mat, src, psi, prec=None):
        # verbosity
        self.verbose = g.default.is_verbose("fgmres")
        checkres = True # for now

        # total time
        tt0 = time()

        # parameters
        rlen = self.restartlen

        # tensors
        dtype = np.complex128
        H = np.zeros((rlen + 1, rlen), dtype)
        c = np.zeros((rlen + 1), dtype)
        s = np.zeros((rlen + 1), dtype)
        y = np.zeros((rlen + 1), dtype)
        gamma = np.zeros((rlen + 1), dtype)

        # fields
        mmpsi, r = g.copy(src), g.copy(src),
        V = [g.lattice(src) for i in range(rlen + 1)]
        if not prec is None: # save vectors if unpreconditioned
            Z = [g.lattice(src) for i in range(rlen + 1)]

        # residual
        ssq = g.norm2(src)
        rsq = self.eps**2. * ssq

        # initial values
        r2 = self.restart(mat, psi, mmpsi, src, r, V, gamma)

        for k in range(self.maxiter):
            # iteration within current krylov space
            i = k % rlen

            # iteration criteria
            reached_maxiter = (k+1 == self.maxiter)
            need_restart = (i+1 == rlen)

            t0 = time()
            if not prec is None:
                prec(V[i], Z[i])
            t1 = time()
            
            t2 = time()
            if not prec is None:
                mat(Z[i], V[i + 1])
            else:
                mat(V[i], V[i + 1])
            t3 = time()

            t4 = time()
            g.orthogonalize(V[i + 1], V[0:i + 1], H[:, i])
            t5 = time()

            t6 = time()
            H[i + 1, i] = g.norm2(V[i + 1])**0.5

            if H[i + 1, i] == 0.:
                g.message("fgmres breakdown, H[%d, %d] = 0" % (i + 1, i))
                break

            V[i + 1] /= H[i + 1, i]
            t7 = time()

            t8 = time()
            self.qr_update(s, c, H, gamma, i)
            r2 = np.absolute(gamma[i + 1])**2
            t9 = time()

            if self.verbose:
                g.message(
                    "Timing[s]: Prec = %g, Matrix = %g, Orthog = %g, RestArnoldi = %g, QR = %g"
                    % (t1 - t0, t3 - t2, t5 - t4, t7 - t6, t9 - t8))
                g.message("res^2[ %d, %d ] = %g" % (k, i, r2))

            if r2 <= rsq or need_restart or reached_maxiter:
                if not prec is None:
                    self.update_psi(psi, gamma, H, y, Z, i)
                else:
                    self.update_psi(psi, gamma, H, y, V, i)

                if r2 <= rsq:
                    if self.verbose:
                        tt1 = time()
                        g.message("Converged in %g s" % (tt1 - tt0))
                    if checkres:
                        res = self.calc_res(mat, psi, mmpsi, src, r) / ssq
                        g.message(
                            "Computed res = %g, true res = %g, target = %g" %
                            (r2**0.5, res**0.5, self.eps))
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
                    r2 = self.restart(mat, psi, mmpsi, src, r, V, gamma)
                    if self.verbose:
                        g.message("Performed restart")

    def qr_update(self, s, c, H, gamma, i):
        # apply previous givens to matrix
        for j in range(i):
            tmp = -s[j] * H[j, i] + c[j] * H[j + 1, i]
            H[j, i] = np.conjugate(c[j]) * H[j, i] + np.conjugate(s[j]) * H[j + 1, i]
            H[j+ 1, i] = tmp

        # compute new rotation matrix
        den = (np.absolute(H[i, i])**2 + np.absolute(H[i + 1, i])**2)**0.5
        c[i] = H[i, i] / den
        s[i] = H[i + 1, i] / den

        # apply new givens to matrix
        H[i, i] = den
        H[i + 1, i] = 0.

        # apply new givens to vector
        gamma[i + 1] = -s[i] * gamma[i]
        gamma[i] *= np.conjugate(c[i])

    def update_psi(self, psi, gamma, H, y, V, i):
        # backward substitution
        for j in reversed(range(i + 1)):
            y[j] = (gamma[j] - np.dot(H[j, j+1:i+1], y[j+1:i+1])) / H[j, j]

        for j in range(i + 1):
            psi += y[j] * V[j]

    def restart(self, mat, psi, mmpsi, src, r, V, gamma):
        r2 = self.calc_res(mat, psi, mmpsi, src, r)
        gamma[0] = r2**0.5
        V[0] @= r / gamma[0]
        return r2

    def calc_res(self, mat, psi, mmpsi, src, r):
        mat(psi, mmpsi)
        return g.axpy_norm2(r, -1., mmpsi, src)
