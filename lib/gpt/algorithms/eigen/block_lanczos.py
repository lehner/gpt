#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


# Implements a variation of https://arxiv.org/pdf/1812.07090
class block_lanczos:
    @g.params_convention(
        orthogonalize_nblock=4,
        Nstop=None,
        resid=None,
        maxiter=None,
        miniter=None,
        step=None,
        lambda_max=None,
    )
    def __init__(self, params):
        self.params = params

    def __call__(self, mat, Y0):
        # verbosity
        verbose = g.default.is_verbose("block_lanczos")
        verbose_performance = g.default.is_verbose("block_lanczos_performance")

        maxiter = self.params["maxiter"]
        miniter = self.params["miniter"]
        nblock = self.params["orthogonalize_nblock"]
        step = self.params["step"]
        Nstop = self.params["Nstop"]
        resid = self.params["resid"]
        lambda_max = self.params["lambda_max"]

        # first approximate largest eigenvalue
        if lambda_max is None:
            g.default.push_verbose("power_iteration_convergence", True)
            pit = g.algorithms.eigen.power_iteration(eps=0.02, maxiter=10, real=True)
            g.default.pop_verbose()
            lambda_max = pit(mat, Y0[0])[0]

        # main
        block_size = len(Y0)
        t = g.timer("block_lanczos")
        Y = Y0
        t("orthonormalize")
        g.orthonormalize(Y, nblock=nblock)
        alpha = []
        beta = []
        for n in range(maxiter):

            t0 = g.time()
            Y_nm = Y[block_size * n : block_size * (n + 1)]
            t("matrix")
            Y_n_tilde = g(mat * g.expr(Y_nm))
            t1 = g.time()

            if n > 0:
                # Original algorithm used beta to orthogonalize, found this to be too imprecise in single-precision
                t("orthonormalize")
                for y in Y_n_tilde:
                    g.orthogonalize(y, Y[0 : block_size * n], nblock=nblock)

            t("orthonormalize")
            alpha_nm = g.inner_product(Y_nm, Y_n_tilde)
            alpha.append(alpha_nm)

            for j, y in enumerate(Y_n_tilde):
                g.linear_combination(y, [y] + Y_nm, [1.0] + [-x for x in alpha_nm[:, j]])

            Y_n = Y_n_tilde
            beta_nm = np.zeros(shape=(block_size, block_size), dtype=np.complex128)
            for j in range(block_size):
                g.orthogonalize(Y_n[j], Y_n_tilde[:j], ips=beta_nm[0:j, j], nblock=nblock)
                c = g.norm2(Y_n[j]) ** 0.5
                beta_nm[j, j] = c
                Y_n[j] /= c

            Y.extend(Y_n)
            beta.append(beta_nm)

            t2 = g.time()
            if verbose:
                g.message(
                    f"block_lanczos iteration {n} took {t1 - t0:g} s for matrix and {t2 - t1:g} s for orthogonalization"
                )

            # check for convergence
            space_size = block_size * (n + 1)
            if n % step == step - 1 and space_size >= max(miniter, Nstop):
                if verbose:
                    g.message(f"block_lanczos iteration {n + 1} x {block_size}")
                t("construct H")
                H_nm = np.zeros(shape=(space_size, space_size), dtype=np.complex128)
                for i in range(n + 1):
                    H_nm[
                        block_size * i : block_size * (i + 1), block_size * i : block_size * (i + 1)
                    ] = alpha[i]
                    if i < n - 1:
                        H_nm[
                            block_size * (i + 1) : block_size * (i + 2),
                            block_size * i : block_size * (i + 1),
                        ] = beta[i]
                        H_nm[
                            block_size * i : block_size * (i + 1),
                            block_size * (i + 1) : block_size * (i + 2),
                        ] = np.conjugate(beta[i].T)

                t("eigh")
                eig_val, eig_vec = np.linalg.eigh(H_nm)

                idx = eig_val.argsort()[::-1]
                eig_val = eig_val[idx]
                eig_vec = eig_vec[:, idx]
                t()

                allconv = True

                jj = 1
                B = g.copy(Y[0])
                while jj <= Nstop:
                    j = Nstop - jj
                    g.linear_combination(B, Y[0:space_size], eig_vec[:, j])
                    B *= 1.0 / g.norm2(B) ** 0.5
                    v = g(mat * B)

                    ev_test = g.inner_product(B, v).real
                    eps2 = g.norm2(v - ev_test * B) / lambda_max**2.0
                    if verbose:
                        g.message(
                            "%-65s %-45s %-50s"
                            % (
                                "ev[ %d ] = %s" % (j, eig_val[j]),
                                "<B|M|B> = %s" % (ev_test),
                                "|M B - ev B|^2 / ev_max^2 = %s" % (eps2),
                            )
                        )
                    if eps2 > resid:
                        allconv = False
                    if jj == Nstop:
                        break
                    jj = min([Nstop, 2 * jj])

                if allconv:
                    if verbose:
                        g.message("Converged in %d iterations" % (n + 1))
                    break

        t("rotate")
        g.rotate(
            Y[0:space_size], np.ascontiguousarray(eig_vec.T), 0, Nstop, 0, space_size
        )
        t()

        if verbose_performance:
            g.message(t)

        return Y[0:Nstop], eig_val[0:Nstop]
