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
from gpt.algorithms import base_iterative


class playback_gcr(base_iterative):
    @g.params_convention(monitor_convergence=None)
    def __init__(self, alphas, params):
        super().__init__()
        self.alphas = alphas
        self.monitor_convergence = params["monitor_convergence"]

    def __call__(self, value):
        alphas = self.alphas

        if g.util.is_num(value):
            y = alphas[0]
            r = 1.0
            for i in range(len(alphas) - 1):
                r -= alphas[i] * value * r
                y += alphas[i + 1] * r
            return y
        else:
            mat = value

            vector_space = None
            if isinstance(mat, g.matrix_operator):
                vector_space = mat.vector_space
                mat = mat.specialized_list_callable()

            @self.timed_function
            def inv(y, src, t):
                t("setup")
                g(y, alphas[0] * g.expr(src))
                r = g.copy(src)
                mat_r = [g.lattice(s) for s in src]
                for i in range(len(alphas) - 1):
                    t("mat")
                    mat(mat_r, r)
                    t("axpy")
                    for j in range(len(src)):
                        g.axpy(r[j], -alphas[i], mat_r[j], r[j])
                        g.axpy(y[j], alphas[i + 1], r[j], y[j])

                    if self.monitor_convergence is not None:
                        if i % self.monitor_convergence == 0:
                            r2 = sum(g.norm2(r))
                            self.log_convergence(i, r2)

            return g.matrix_operator(
                mat=inv,
                inv_mat=mat,
                vector_space=vector_space,
                accept_guess=False,
                accept_list=True,
            )


class recording_gcr(base_iterative):

    @g.params_convention(eps=1e-15, maxiter=1000000)
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.alphas = []

    def modified(self, **params):
        return recording_gcr({**self.params, **params})

    def __call__(self, mat):
        vector_space = None
        if isinstance(mat, g.matrix_operator):
            vector_space = mat.vector_space

        @self.timed_function
        def inv(psi, src, t):
            t("setup")

            # tensors
            alpha = 0.0

            # initial residual
            psi[:] = 0
            r = g.copy(src)
            r2 = g.norm2(r)

            # source
            ssq = g.norm2(src)
            if ssq == 0.0:
                assert r2 != 0.0
                ssq = r2

            # target residual
            rsq = self.eps**2.0 * ssq

            self.alphas.clear()
            for k in range(0, self.maxiter):
                t("mat")
                mat_r = g(mat * r)

                t("inner_product")
                A, rhs = g.inner_product(mat_r, [mat_r, r])[0]

                t("solve")
                alpha = rhs / A
                self.alphas.append(alpha)

                t("update_psi")
                psi += alpha * r

                t("update_residual")
                r -= alpha * mat_r

                t("residual")
                r2 = g.norm2(r)

                t("other")
                self.log_convergence(k, r2, rsq)

                if r2 <= rsq:
                    msg = f"converged in {k + 1} iterations"
                    msg += f";  computed squared residual {r2:e} / {rsq:e}"
                    self.log(msg)
                    return

            msg = f"NOT converged in {k + 1} iterations"
            msg += f";  computed squared residual {r2:e} / {rsq:e}"
            self.log(msg)

        return g.matrix_operator(
            mat=inv, inv_mat=mat, vector_space=vector_space, accept_guess=False
        )
