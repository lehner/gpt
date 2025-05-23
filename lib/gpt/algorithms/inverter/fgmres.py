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


class fgmres(base_iterative):
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
        return fgmres({**self.params, **params})

    def qr_update(self, s, c, H, gamma, i):
        # apply previous givens to matrix
        for j in range(i):
            tmp = -s[j] * H[j, i] + c[j] * H[j + 1, i]
            H[j, i] = np.conjugate(c[j]) * H[j, i] + np.conjugate(s[j]) * H[j + 1, i]
            H[j + 1, i] = tmp

        # compute new rotation matrix
        den = (np.absolute(H[i, i]) ** 2 + np.absolute(H[i + 1, i]) ** 2) ** 0.5
        c[i] = H[i, i] / den
        s[i] = H[i + 1, i] / den

        # apply new givens to matrix
        H[i, i] = den
        H[i + 1, i] = 0.0

        # apply new givens to vector
        gamma[i + 1] = -s[i] * gamma[i]
        gamma[i] *= np.conjugate(c[i])

    def update_psi(self, psi, gamma, H, y, V, i):
        # backward substitution
        for l in range(len(psi)):
            for j in reversed(range(i + 1)):
                y[j, l] = (gamma[j, l] - np.dot(H[j, j + 1 : i + 1, l], y[j + 1 : i + 1, l])) / H[j, j, l]

            for j in range(i + 1):
                psi[l] += y[j, l] * V[j][l]

    def restart(self, mat, psi, mmpsi, src, r, V, Z, gamma, t):
        r2 = self.calc_res(mat, psi, mmpsi, src, r, t)
        t("restart - misc")
        gamma[0] = np.array(r2)**0.5
        for i in range(len(src)):
            g.axpy(V[0][i], 1.0 / gamma[0][i] - 1.0, r[i], r[i])
        t("restart - zero")
        if Z is not None:
            for z in Z:
                for z_i in z:
                    z_i[:] = 0
        return sum(r2)

    def calc_res(self, mat, psi, mmpsi, src, r, t):
        t("mat")
        mat(mmpsi, psi)
        t("axpy")
        for r_i, mmpsi_i, src_i in zip(r, mmpsi, src):
            g.axpy(r_i, -1.0, mmpsi_i, src_i)
        t("norm2")
        return g.norm2(r)

    def __call__(self, mat):
        vector_space = None

        prec = self.prec(mat) if self.prec is not None else None

        if isinstance(mat, g.matrix_operator):
            vector_space = mat.vector_space
            mat = mat.specialized_list_callable()

        if isinstance(prec, g.matrix_operator):
            prec = prec.specialized_list_callable()

        @self.timed_function
        def inv(psi, src, t):
            # timing
            t("setup")

            # parameters
            n_rhs = len(src)
            rlen = self.restartlen

            # tensors
            dtype = g.double.complex_dtype
            H = np.zeros((rlen + 1, rlen, n_rhs), dtype)
            c = np.zeros((rlen + 1, n_rhs), dtype)
            s = np.zeros((rlen + 1, n_rhs), dtype)
            y = np.zeros((rlen + 1, n_rhs), dtype)
            gamma = np.zeros((rlen + 1, n_rhs), dtype)

            # fields
            mmpsi, r = (
                g.copy(src),
                g.copy(src),
            )
            V = [[g.lattice(s) for s in src] for i in range(rlen + 1)]
            Z = (
                [[g.lattice(s) for s in src] for i in range(rlen + 1)] if prec is not None else None
            )  # save vectors if unpreconditioned
            ZV = Z if prec is not None else V

            # initial residual
            t("restart")
            r2 = self.restart(mat, psi, mmpsi, src, r, V, Z, gamma, t)
            t("setup")

            # source
            ssq = sum(g.norm2(src))
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
                    for z in ZV[i]:
                        z[:] = 0
                    prec(ZV[i], V[i])

                t("mat")
                mat(V[i + 1], ZV[i])

                t("ortho")
                for j in range(n_rhs):
                    g.orthogonalize(V[i + 1][j], [V[l][j] for l in range(0, i + 1)], H[:, i, j], nblock=10)

                t("linalg norm2")
                H[i + 1, i] = np.array(g.norm2(V[i + 1])) ** 0.5
                for j in range(n_rhs):
                    assert H[i + 1, i, j] != 0.0
                t("linalg div")
                for j in range(n_rhs):
                    V[i + 1][j] /= H[i + 1, i, j]

                t("qr")
                self.qr_update(s, c, H, gamma, i)

                t("other")
                r2 = np.sum(np.absolute(gamma[i + 1]) ** 2)
                self.log_convergence((k, i), r2, rsq)

                if r2 <= rsq or need_restart:
                    t("update_psi")
                    self.update_psi(psi, gamma, H, y, ZV, i)

                if r2 <= rsq:
                    msg = f"converged in {k + 1} iterations;  computed squared residual {r2:e} / {rsq:e}"
                    if self.checkres:
                        res = sum(self.calc_res(mat, psi, mmpsi, src, r, t))
                        msg += f";  true squared residual {res:e} / {rsq:e}"
                    self.log(msg)
                    return

                if need_restart:
                    t("restart")
                    r2 = self.restart(mat, psi, mmpsi, src, r, V, Z, gamma, t)
                    self.debug("performed restart")

            msg = f"NOT converged in {k + 1} iterations;  computed squared residual {r2:e} / {rsq:e}"
            if self.checkres:
                res = sum(self.calc_res(mat, psi, mmpsi, src, r, t))
                msg += f";  true squared residual {res:e} / {rsq:e}"
            self.log(msg)

        return g.matrix_operator(
            mat=inv,
            inv_mat=mat,
            vector_space=vector_space,
            accept_guess=(True, False),
            accept_list=True
        )
