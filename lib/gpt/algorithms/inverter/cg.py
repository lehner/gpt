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
    @g.params_convention(eps=1e-15, maxiter=1000000)
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]

    def __call__(self, mat, approx_inv_mat=None):

        # remove wrapper for performance benefits
        otype, grid, cb = None, None, None
        if type(mat) == g.matrix_operator:
            otype, grid, cb = mat.otype, mat.grid, mat.cb
            mat = mat.mat
        if type(approx_inv_mat) == g.matrix_operator:
            approx_inv_mat = approx_inv_mat.mat

        @self.timed_function
        def inv(psi, src, t):
            assert src != psi
            t("setup")
            p, mmp, r = g.copy(src), g.copy(src), g.copy(src)
            if approx_inv_mat is not None:
                z = g.copy(src)
            else:
                z = r  # reference the same lattice object
            mat(mmp, psi)  # in, out
            d = g.inner_product(psi, mmp).real
            b = g.norm2(mmp)
            r @= src - mmp
            if approx_inv_mat is not None:
                approx_inv_mat(z, r)
            p @= z
            cp = g.inner_product(r, z).real  # this is always real
            ssq = g.norm2(src)
            if ssq == 0.0:
                assert cp != 0.0  # need either source or psi to not be zero
                ssq = cp
            rsq = self.eps ** 2.0 * ssq
            for k in range(self.maxiter):
                c = cp

                t("matrix")
                mat(mmp, p)

                t("inner_product")
                d = g.inner_product(p, mmp).real
                a = c / d

                t("axpy_norm2")
                norm_r = g.axpy_norm2(r, -a, mmp, r)
                if approx_inv_mat is None:
                    cp = norm_r
                else:
                    t("preconditioner")
                    approx_inv_mat(z, r)
                    cp = g.inner_product(r, z).real

                t("linear combination")
                b = cp / c
                psi += a * p
                p @= b * p + z

                t("other")
                self.log_convergence(k, norm_r, rsq)
                if norm_r <= rsq:
                    self.log(f"converged in {k+1} iterations")
                    return

            self.log(
                f"NOT converged in {k+1} iterations;  squared residual {norm_r:e} / {rsq:e}"
            )

        return g.matrix_operator(
            mat=inv,
            inv_mat=mat,
            otype=otype,
            accept_guess=(True, False),
            grid=grid,
            cb=cb,
        )
