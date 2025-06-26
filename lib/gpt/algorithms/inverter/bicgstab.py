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
from gpt.algorithms import base_iterative


class bicgstab(base_iterative):
    @g.params_convention(eps=1e-15, maxiter=1000000, r0hat=None)
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.r0hat = params["r0hat"]

    def __call__(self, mat):
        vector_space = None
        if isinstance(mat, g.matrix_operator):
            vector_space = mat.vector_space

        @self.timed_function
        def inv(psi, src, time):
            time("setup")

            r2 = g.norm2(psi)
            ssq = g.norm2(src)
            if ssq == 0.0:
                assert r2 != 0.0  # need either source or psi to not be zero
                ssq = r2
            rsq = self.eps**2.0 * ssq

            r = g(src - mat * psi)
            rhat = g.copy(r) if self.r0hat is None else self.r0hat
            rho = 1
            alpha = 1
            omega = 1
            p = g(0.0 * r)
            v = g(0.0 * r)

            for k in range(self.maxiter):
                rho_prev = rho

                time("inner_product")
                Crho = g.inner_product(rhat, r)
                rho = Crho.real
                beta = (rho / rho_prev) * (alpha / omega)
                bo = beta * omega

                time("linear algebra")
                p = g(beta * p - bo * v + r)

                time("matrix")
                v = g(mat * p)

                time("inner_product")
                Calpha = g.inner_product(rhat, v)
                if Calpha.real == 0.0:
                    time("restart")
                    rho = 1
                    alpha = 1
                    omega = 1
                    v[:] = 0
                    p[:] = 0
                    r = g(src - mat * psi)
                    g.message("Restart due to numerical instability")
                    continue

                alpha = rho / Calpha.real

                time("linear algebra")
                h = g(alpha * p + psi)
                s = g(-alpha * v + r)

                time("matrix")
                t = g(mat * s)

                time("inner_product")
                Comega = g.inner_product(t, s)
                omega = Comega.real / g.norm2(t)

                time("linear_algebra")
                psi @= g(h + omega * s)
                r = g(-omega * t + s)

                time("inner_product")
                r2 = g.norm2(r)

                time()

                if r2 <= rsq:
                    self.log(f"converged in {k + 1} iterations")
                    return

                self.log_convergence(k, r2, rsq)

            self.log(f"NOT converged in {k + 1} iterations;  squared residual {r2:e} / {rsq:e}")

        return g.matrix_operator(
            mat=inv,
            inv_mat=mat,
            vector_space=vector_space,
            accept_guess=(True, False),
        )
