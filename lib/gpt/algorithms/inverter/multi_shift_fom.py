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


class shifted_fom:
    def __init__(self, psi, s):
        self.s = s
        self.x = psi
        self.x[:] = 0
        self.rho = 1.0
        self.r2 = 1.0
        self.y = []
        self.converged = False

    def solve_hessenberg(self, H, r2):
        Hs = [h.copy() for h in H]
        for h in Hs:
            h[-2] += self.s

        n = len(Hs)
        b = np.zeros(n, np.complex128)
        b[0] = self.rho * r2**0.5
        for i in range(n - 1):
            k = -Hs[i][-1] / Hs[i][-2]
            for j in range(n - i):
                Hs[i + j][i + 1] += k * Hs[i + j][i]
            b[i + 1] += k * b[i]

        self.y = np.zeros(n, np.complex128)
        for i in reversed(range(n)):
            self.y[i] = b[i] / Hs[i][i]
            for j, hj in enumerate(Hs[i][0:-1]):
                b[j] -= hj * self.y[i]
        self.rho = -Hs[-1][-1] * self.y[-1]
        self.r2 = np.abs(self.rho) ** 2.0

    def update_psi(self, mmp, V):
        g.linear_combination(mmp, V[0:-1], self.y)
        self.x += mmp

    def check(self, rsq):
        if not self.converged:
            if self.r2 <= rsq:
                self.converged = True
                return f"shift {self.s} converged"
        return None

    def calc_res(self, mat, src, mmp):
        mat(mmp, self.x)
        res = g.norm2(src - mmp - self.s * self.x)
        return res


class multi_shift_fom(base_iterative):
    @g.params_convention(
        eps=1e-15,
        maxiter=1000,
        restartlen=20,
        shifts=[],
        checkres=True,
        rhos=False,
    )
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.restartlen = params["restartlen"]
        self.shifts = params["shifts"]
        self.checkres = params["checkres"]
        self.rhos = params["rhos"]

    def arnoldi(self, mat, V, rlen):
        H = []
        for i in range(rlen):
            ips = np.zeros(i + 2, np.complex128)
            mat(V[i + 1], V[i])
            g.orthogonalize(
                V[i + 1],
                V[0 : i + 1],
                ips[0:-1],
                nblock=10,
            )
            ips[-1] = g.norm2(V[i + 1]) ** 0.5
            V[i + 1] /= ips[-1]
            H.append(ips)
        return H

    def restart(self, V):
        V[0] @= g.copy(V[-1])
        r2 = g.norm2(V[0])
        V[0] /= r2**0.5
        return r2

    def __call__(self, mat):
        vector_space = None
        if isinstance(mat, g.matrix_operator):
            vector_space = mat.vector_space
            mat = mat.mat
            # remove wrapper for performance benefits

        @self.timed_function
        def inv(psi, src, t):
            if len(src) > 1:
                n = len(src)
                # do different sources separately
                for idx in range(n):
                    inv(psi[idx::n], [src[idx]])
                return

            # timing
            t("setup")

            # fields
            src = src[0]
            mmp = g.copy(src)

            # initial residual
            r2 = g.norm2(src)
            assert r2 != 0.0

            # target residual
            rsq = self.eps**2.0 * r2

            # restartlen
            rlen = self.restartlen

            # shifted systems
            sfoms = []
            for j, s in enumerate(self.shifts):
                sfoms += [shifted_fom(psi[j], s)]

            # krylov space
            V = [g.copy(src) for i in range(rlen + 1)]
            V[0] /= r2**0.5

            # return rhos for prec fgmres
            rr = self.rhos

            for k in range(0, self.maxiter, rlen):
                t("arnoldi")
                H = self.arnoldi(mat, V, rlen)

                for j, fom in enumerate(sfoms):
                    if fom.converged is False or rr:
                        t("solve_hessenberg")
                        fom.solve_hessenberg(H, r2)

                        t("update_psi")
                        fom.update_psi(mmp, V)

                        t("other")
                        self.log_convergence((k, j), fom.r2, rsq)

                t("other")
                for fom in sfoms:
                    msg = fom.check(rsq)
                    if msg:
                        msg += f" at iteration {k+rlen}"
                        if self.maxiter != rlen:
                            msg += f";  computed squared residual {fom.r2:e} / {rsq:e}"
                        if self.checkres:
                            res = fom.calc_res(mat, src, mmp)
                            msg += f";  true squared residual {res:e} / {rsq:e}"
                        self.log(msg)

                if all([fom.converged for fom in sfoms]):
                    self.log(f"converged in {k+rlen} iterations")
                    return [fom.rho for fom in sfoms] if rr else None

                if self.maxiter != rlen:
                    t("restart")
                    r2 = self.restart(V)
                    self.debug("performed restart")

            t("other")
            for fom in sfoms:
                if fom.converged is False:
                    msg = f"shift {fom.s} NOT converged in {k+rlen} iterations"
                    if self.maxiter != rlen:
                        msg += f";  computed squared residual {fom.r2:e} / {rsq:e}"
                    if self.checkres:
                        res = fom.calc_res(mat, src, mmp)
                        msg += f";  true squared residual {res:e} / {rsq:e}"
                    self.log(msg)
            cs = sum([fom.converged for fom in sfoms if True])
            ns = len(self.shifts)
            self.log(f"NOT converged in {k+rlen} iterations; {cs} / {ns} converged shifts")
            return [fom.rho for fom in sfoms] if rr else None

        return g.matrix_operator(
            mat=inv,
            vector_space=vector_space,
            accept_guess=(True, False),
            accept_list=lambda src: len(src) * len(self.shifts),
        )
