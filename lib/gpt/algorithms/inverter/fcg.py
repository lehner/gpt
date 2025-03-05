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


# Yvan Notay: Flexible Conjugate Gradients
# https://scispace.com/pdf/flexible-conjugate-gradients-1g99nh6qak.pdf
class fcg(base_iterative):
    @g.params_convention(
        eps=1e-15, maxiter=1000000, eps_abs=None, miniter=0, restartlen=10, prec=None
    )
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.eps_abs = params["eps_abs"]
        self.maxiter = params["maxiter"]
        self.restartlen = params["restartlen"]
        self.prec = params["prec"]
        # fcg only makes sense with preconditioner
        assert self.prec is not None

    def modified(self, **params):
        return fcg({**self.params, **params})

    def __call__(self, A):
        prec = self.prec(A)

        vector_space = None
        if isinstance(A, g.matrix_operator):
            vector_space = A.vector_space

        @self.timed_function
        def inv(x, b, t):

            n = len(b)

            t("setup")
            b_norm2 = sum(g.norm2(b))
            if b_norm2 == 0.0:
                for psi in x:
                    psi[:] = 0
                return
            r_norm2 = self.eps**2.0 * b_norm2

            t("matrix")
            r = g(g.expr(b) - A * g.expr(x))

            Ad = {}
            d = {}
            dAd = {}

            for i in range(self.maxiter):
                t("prec")
                w = g(prec * g.expr(r))

                t("misc")
                mi = 0 if i == 0 else max(1, i % self.restartlen)
                # purge history
                keys = list(Ad.keys())
                for j in keys:
                    if j < i - mi:
                        del Ad[j]
                        del d[j]
                        del dAd[j]

                # orthogonalize
                t("orthogonalize")
                d[i] = g.copy(w)
                for l in range(n):
                    for k in range(i - mi, i):
                        d[i][l] -= g.inner_product(w[l], Ad[k][l]) / dAd[k][l] * d[k][l]

                # new components
                t("matrix")
                Ad[i] = g(A * g.expr(d[i]))

                t("inner_product")
                dAd[i] = [g.inner_product(d[i][l], Ad[i][l]) for l in range(n)]

                # update vectors
                for l in range(n):
                    alpha = g.inner_product(d[i][l], r[l]) / dAd[i][l]
                    g.axpy(x[l], alpha, d[i][l], x[l])
                    g.axpy(r[l], -alpha, Ad[i][l], r[l])

                res = sum(g.norm2(r))
                self.log_convergence(i, res, r_norm2)

                if self.eps_abs is not None and res <= self.eps_abs**2.0:
                    self.log(f"converged in {i + 1} iterations (absolute criterion)")
                    return
                if res <= r_norm2:
                    self.log(f"converged in {i + 1} iterations")
                    return

            self.log(
                f"NOT converged in {i + 1} iterations;  squared residual {res:e} / {r_norm2:e}"
            )

        return g.matrix_operator(
            mat=inv,
            inv_mat=A,
            accept_guess=(True, False),
            accept_list=True,
            vector_space=vector_space,
        )
