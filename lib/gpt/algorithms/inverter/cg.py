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
from gpt.algorithms import base_iterative


class cg(base_iterative):
    @g.params_convention(eps=1e-15, maxiter=1000000, eps_abs=None, miniter=0, prec=None)
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.eps_abs = params["eps_abs"]
        self.maxiter = params["maxiter"]
        self.miniter = params["miniter"]
        self.prec = params["prec"]

    def modified(self, **params):
        return cg({**self.params, **params})

    def __call__(self, mat):
        prec = self.prec(mat) if self.prec is not None else None

        vector_space = None
        if isinstance(mat, g.matrix_operator):
            vector_space = mat.vector_space
            mat = mat.specialized_singlet_callable()

        @self.timed_function
        def inv(psi, src, t):
            assert src != psi
            t("setup")
            p, mmp, r = g.lattice(src), g.lattice(src), g.lattice(src)
            if prec is not None:
                z = g.lattice(src)
            t("matrix")
            mat(mmp, psi)  # in, out
            t("setup")
            g.axpy(r, -1.0, mmp, src)
            if prec is not None:
                z[:] = 0
                prec(z, r)
                g.copy(p, z)
                cp = g.inner_product(r, z).real
            else:
                g.copy(p, r)
                cp = g.norm2(p)
            ssq = g.norm2(src)
            if ssq == 0.0:
                psi[:] = 0
                return
            rsq = self.eps**2.0 * ssq
            for k in range(self.maxiter):
                c = cp
                t("matrix")
                mat(mmp, p)
                t("inner_product")
                d = g.inner_product(p, mmp).real
                a = c / d
                t("axpy_norm2")
                if prec is not None:
                    # c = <r,z>, d = <p,A p>
                    g.axpy(r, -a, mmp, r)
                    t("prec")
                    z[:] = 0
                    prec(z, r)
                    t("axpy_norm2")
                    cp = g.inner_product(r, z).real
                else:
                    cp = g.axpy_norm2(r, -a, mmp, r)
                t("linear combination")
                b = cp / c
                psi += a * p
                if prec is not None:
                    g.axpy(p, b, p, z)
                else:
                    g.axpy(p, b, p, r)
                t("other")
                res = abs(cp)
                self.log_convergence(k, res, rsq)
                if k + 1 >= self.miniter:
                    if self.eps_abs is not None and res <= self.eps_abs**2.0:
                        self.log(f"converged in {k+1} iterations (absolute criterion)")
                        return
                    if res <= rsq:
                        self.log(f"converged in {k+1} iterations")
                        return

            self.log(f"NOT converged in {k+1} iterations;  squared residual {res:e} / {rsq:e}")

        return g.matrix_operator(
            mat=inv, inv_mat=mat, accept_guess=(True, False), vector_space=vector_space
        )
