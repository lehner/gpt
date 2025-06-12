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
from gpt.algorithms import base_iterative


class chebyshev(base_iterative):
    @g.params_convention(low=None, high=None, eps=1e-15, maxiter=1000000, eps_abs=None)
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.eps_abs = params["eps_abs"]
        self.maxiter = params["maxiter"]
        self.low = params["low"]
        self.high = params["high"]
        assert self.low is not None and self.high is not None

    def modified(self, **params):
        return chebyshev({**self.params, **params})

    def __call__(self, A):

        vector_space = None
        if isinstance(A, g.matrix_operator):
            vector_space = A.vector_space

        @self.timed_function
        def inv(x, b, t):
            t("setup")
            d = (self.high + self.low) / 2
            c = (self.high - self.low) / 2
            t("matrix")
            r = g(g.expr(b) - A * g.expr(x))
            t("setup")
            b_norm2 = sum(g.norm2(b))
            if b_norm2 == 0.0:
                for psi in x:
                    psi[:] = 0
                return
            r_norm2 = self.eps**2.0 * b_norm2
            for k in range(self.maxiter):

                t("linear algebra")
                if k == 0:
                    alpha = 1 / d
                    p = r
                elif k == 1:
                    beta = (1 / 2) * (c * alpha) ** 2
                    alpha = 1 / (d - beta / alpha)
                    p = g(g.expr(r) + beta * g.expr(p))
                else:
                    beta = (c * alpha / 2) ** 2
                    alpha = 1 / (d - beta / alpha)
                    p = g(g.expr(r) + beta * g.expr(p))

                g(x, g.expr(x) + alpha * g.expr(p))
                t("matrix")
                r = g(g.expr(b) - A * g.expr(x))
                t("norm2")
                res = sum(g.norm2(r))
                self.log_convergence(k, res, r_norm2)

                if self.eps_abs is not None and res <= self.eps_abs**2.0:
                    self.log(f"converged in {k + 1} iterations (absolute criterion)")
                    return
                if res <= r_norm2:
                    self.log(f"converged in {k + 1} iterations")
                    return

            self.log(
                f"NOT converged in {k + 1} iterations;  squared residual {res:e} / {r_norm2:e}"
            )

        return g.matrix_operator(
            mat=inv,
            inv_mat=A,
            accept_guess=(True, False),
            accept_list=True,
            vector_space=vector_space,
        )
