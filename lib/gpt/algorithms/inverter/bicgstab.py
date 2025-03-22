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
    @g.params_convention(eps=1e-15, maxiter=1000000)
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]

    def __call__(self, mat):
        vector_space = None
        if isinstance(mat, g.matrix_operator):
            vector_space = mat.vector_space
            mat = mat.specialized_singlet_callable()

        @self.timed_function
        def inv(x, b, t):
            t("setup")

            r = g(b - mat * x)
            rhat = r
            rho = g.inner_product(rhat, r)
            p = r

            r2 = g.norm2(x)
            ssq = g.norm2(b)
            if ssq == 0.0:
                assert r2 != 0.0  # need either source or psi to not be zero
                ssq = r2
            rsq = self.eps**2.0 * ssq

            for k in range(self.maxiter):
                t("mat")
                nu = g(mat * p)
                t("linear algebra")
                alpha = rho/g.inner_product(rhat, nu)
                h = g(x + alpha * p)
                s = g(r - alpha * nu)
                res1 = g.norm2(s)
                if res1 <= rsq:
                    x @= h
                    self.log(f"converged in {k+1} iterations")
                    break

                t = g(mat * s)
                omega = g.inner_product(t,s) / g.norm2(t)
                x @= h + omega * s
                r @= s - omega * t
                res2 = g.norm2(r)
                if res2 <= rsq:
                    self.log(f"converged in {k+1} iterations")
                    break

                rhoprev = rho
                rho = g.inner_product(rhat, r)
                beta = (rho / rhoprev) * (alpha/omega)
                p @= r + beta * p - (beta*omega) * nu

                g.message(k, res1, res2)
                r2 = min(res1,res2)
                self.log_convergence(k, r2, rsq)

                if r2 <= rsq:
                    self.log(f"converged in {k+1} iterations")
                    return

            self.log(f"NOT converged in {k+1} iterations;  squared residual {r2:e} / {rsq:e}")

        return g.matrix_operator(
            mat=inv,
            inv_mat=mat,
            vector_space=vector_space,
            accept_guess=(True, False),
        )
