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


class shifted_fgmres:
    def __init__(self, psi, src, s, rlen, prec):
        self.s = s
        self.x = psi
        self.x[:] = 0
        self.Z = [g.copy(src) for i in range(rlen)] if prec is not None else None
        self.rho = 1.0
        self.gamma = np.zeros(rlen, np.complex128)
        self.r2 = 1.0
        self.y = []
        self.converged = False

    def hessenberg(self, H, prec):
        Hs = [h.copy() for h in H]
        for h, ga in zip(Hs, self.gamma):
            if prec is not None:
                h *= ga
                h[-2] -= ga - 1.0
            else:
                h[-2] += self.s
        return Hs

    def solve_hessenberg(self, Hs, r2, r2_new):
        n = len(Hs)
        b = np.zeros(n, np.complex128)
        b[0] = self.rho * r2**0.5
        for i in range(n - 1):
            k = -Hs[i][-1] / Hs[i][-2]
            for j in range(n - i):
                Hs[i + j][i + 1] += k * Hs[i + j][i]
            b[i + 1] += k * b[i]

        y = np.zeros(n, np.complex128)
        for i in reversed(range(n)):
            y[i] = b[i] / Hs[i][i]
            for j, hj in enumerate(Hs[i][0:-1]):
                b[j] -= hj * y[i]

        self.y = y[0:-1]
        self.rho = y[-1]
        self.r2 = np.abs(self.rho) ** 2.0 * r2_new

    def qr(self, Hs, r2):
        n = len(Hs)
        b = np.zeros(n + 1, np.complex128)
        b[0] = r2**0.5
        for i in range(n):
            den = (Hs[i][-1] ** 2.0 + Hs[i][-2] ** 2.0) ** 0.5
            s = Hs[i][-1] / den
            c = Hs[i][-2] / den
            for j in range(n - i):
                tmp = -s * Hs[j + i][i] + c * Hs[j + i][i + 1]
                Hs[j + i][i] = c * Hs[j + i][i] + s * Hs[j + i][i + 1]
                Hs[j + i][i + 1] = tmp
            tmp = -s * b[i] + c * b[i + 1]
            b[i] = c * b[i] + s * b[i + 1]
            b[i + 1] = tmp

        self.y = np.zeros(n, np.complex128)
        for i in reversed(range(n)):
            self.y[i] = b[i] / Hs[i][i]
            for j, hj in enumerate(Hs[i][0:-1]):
                b[j] -= hj * self.y[i]
        self.r2 = np.abs(b[-1]) ** 2.0

    def update_psi(self, mmp, V, prec):
        ZV = self.Z if prec is not None else V[0:-1]
        g.linear_combination(mmp, ZV, self.y)
        self.x += mmp

    def update_res(self, mat, r, src, mmp):
        mat(mmp, self.x)
        r @= g.eval(src - mmp - self.s * self.x)

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


class multi_shift_fgmres(base_iterative):
    @g.params_convention(
        eps=1e-15,
        restartlen=20,
        maxiter=1000,
        shifts=[],
        checkres=True,
        prec=None,
        rhos=False,
    )
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.restartlen = params["restartlen"]
        self.maxiter = params["maxiter"]
        self.shifts = params["shifts"]
        self.checkres = params["checkres"]
        self.prec = params["prec"]
        self.rhos = params["rhos"]

    def arnoldi(self, mat, V, rlen, mmp, sfgmres, prec, idx, t):
        H = []
        for i in range(rlen):
            if prec is not None:
                t("prec")
                Z = [sfgmres[j].Z[i] for j in idx] + [mmp]
                for z in Z:
                    z[:] = 0
                rhos = prec(Z, [V[i]])
                for j, rho in zip(idx, rhos[0:-1]):
                    sfgmres[j].gamma[i] = rho / rhos[-1]

            t("arnoldi")
            ips = np.zeros(i + 2, np.complex128)
            zv = mmp if prec is not None else V[i]
            mat(V[i + 1], zv)
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

    def setup_prec(self, mat):
        if self.prec is not None:
            prec_shifts = self.shifts + [0.0]
            self.prec.shifts = prec_shifts
            self.prec.rhos = True
            return self.prec(mat).mat
        return None

    def restart_prec(self, mat, sfgmres):
        shifts_prec = [self.shifts[0]]
        idx = [0]
        for j, fgmres in enumerate(sfgmres[1:]):
            if fgmres.converged is False:
                shifts_prec += [fgmres.s]
                idx += [j + 1]
        shifts_prec += [0.0]
        self.prec.shifts = shifts_prec
        prec = self.prec(mat).mat
        return prec, idx

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
            mmp, r = g.copy(src), g.copy(src)

            # initial residual
            r2 = g.norm2(src)
            assert r2 != 0.0

            # target residual
            rsq = self.eps**2.0 * r2

            # restartlen
            rlen = self.restartlen

            # prec
            prec = self.setup_prec(mat)
            plen = len(self.shifts)
            idx = [i for i in range(plen)]

            # shifted systems
            sfgmres = []
            for j, s in enumerate(self.shifts):
                sfgmres += [shifted_fgmres(psi[j], src, s, rlen, prec)]

            # krylov space
            V = [g.copy(src) for i in range(rlen + 1)]
            V[0] /= r2**0.5

            # return rhos for prec fgmres
            rr = self.rhos

            for k in range(0, self.maxiter, rlen):
                # arnoldi
                H = self.arnoldi(mat, V, rlen, mmp, sfgmres, prec, idx, t)

                t("hessenberg")
                fgmres = sfgmres[0]
                Hs = fgmres.hessenberg(H, prec)

                t("qr")
                fgmres.qr(Hs, r2)

                t("update_psi")
                fgmres.update_psi(mmp, V, prec)

                t("update_res")
                r2_new = fgmres.r2
                fgmres.update_res(mat, r, src, mmp)

                t("inner_product")
                vr = [g.inner_product(v, r) for v in V]

                t("other")
                self.log_convergence((k, 0), r2_new, rsq)

                for j, fgmres in enumerate(sfgmres[1:]):
                    if fgmres.converged is False or rr:
                        t("hessenberg")
                        Hs = fgmres.hessenberg(H, prec)
                        Hs.append(vr.copy())

                        t("solve_hessenberg")
                        fgmres.solve_hessenberg(Hs, r2, r2_new)

                        t("update_psi")
                        fgmres.update_psi(mmp, V, prec)

                        t("other")
                        self.log_convergence((k, j + 1), fgmres.r2, rsq)

                t("other")
                for fgmres in sfgmres:
                    msg = fgmres.check(rsq)
                    if msg:
                        msg += f" at iteration {k+rlen}"
                        if self.maxiter != rlen:
                            msg += f";  computed squared residual {fgmres.r2:e} / {rsq:e}"
                        if self.checkres:
                            res = fgmres.calc_res(mat, src, mmp)
                            msg += f";  true squared residual {res:e} / {rsq:e}"
                        self.log(msg)

                if all([fgmres.converged for fgmres in sfgmres]):
                    self.log(f"converged in {k+rlen} iterations")
                    return [fgmres.rho for fgmres in sfgmres] if rr else None

                if self.maxiter != rlen:
                    t("restart")
                    r2 = g.norm2(r)
                    V[0] @= r / r2**0.5

                    if prec is not None and rr is False:
                        t("restart_prec")
                        plen_new = sum([not fgmres.converged for fgmres in sfgmres[1:]]) + 1
                        if plen_new != plen:
                            plen = plen_new
                            prec, idx = self.restart_prec(mat, sfgmres)
                    self.debug("performed restart")

            t("other")
            for fgmres in sfgmres:
                if fgmres.converged is False:
                    msg = f"shift {fgmres.s} NOT converged in {k+rlen} iterations"
                    if self.maxiter != rlen:
                        msg += f";  computed squared residual {fgmres.r2:e} / {rsq:e}"
                    if self.checkres:
                        res = fgmres.calc_res(mat, src, mmp)
                        msg += f";  true squared residual {res:e} / {rsq:e}"
                    self.log(msg)
            cs = sum([fgmres.converged for fgmres in sfgmres if True])
            ns = len(self.shifts)
            self.log(f"NOT converged in {k+rlen} iterations; {cs} / {ns} converged shifts")
            return [fgmres.rho for fgmres in sfgmres] if rr else None

        return g.matrix_operator(
            mat=inv,
            vector_space=vector_space,
            accept_guess=(True, False),
            accept_list=lambda src: len(src) * len(self.shifts),
        )
