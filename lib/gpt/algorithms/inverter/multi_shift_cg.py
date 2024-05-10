#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2022  Mattia Bruno
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


class shifted_cg:
    def __init__(self, psi, src, s):
        self.s = s
        self.p = g.copy(src)
        self.x = psi
        self.x[:] = 0
        self.a = 1.0
        self.rh = 1.0
        self.gh = 1.0
        self.b = 0.0
        self.converged = False

    def step1(self, a, b, om):
        rh = 1.0 / (1.0 + self.s * a + (1.0 - self.rh) * om)
        self.rh = rh
        self.a = rh * a
        self.b = rh**2 * b
        self.gh = self.gh * rh

    def step2(self, r):
        self.x += self.a * self.p
        self.p @= self.b * self.p + self.gh * r

    def check(self, cp, rsq):
        if not self.converged:
            if self.gh**2.0 * cp <= rsq:
                self.converged = True
                return f"shift {self.s} converged"
        return None


class multi_shift_cg(base_iterative):
    @g.params_convention(eps=1e-15, maxiter=1000000, shifts=[])
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.shifts = params["shifts"]

    def __call__(self, mat):
        ns = len(self.shifts)

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

            src = src[0]
            scgs = []
            for j, s in enumerate(self.shifts):
                scgs += [shifted_cg(psi[j], src, s)]

            t("setup")
            p, mmp, r = g.copy(src), g.copy(src), g.copy(src)
            x = g.copy(src)
            x[:] = 0

            b = 0.0
            a = g.norm2(p)
            cp = a
            assert a != 0.0  # need either source or psi to not be zero
            rsq = self.eps**2.0 * a
            for k in range(self.maxiter):
                c = cp
                t("matrix")
                mat(mmp, p)

                t("inner_product")
                dc = g.inner_product(p, mmp)
                d = dc.real
                om = b / a
                a = c / d
                om *= a

                t("axpy_norm2")
                cp = g.axpy_norm2(r, -a, mmp, r)

                t("linear combination")
                b = cp / c
                for cg in scgs:
                    if not cg.converged:
                        cg.step1(a, b, om)

                x += a * p
                p @= b * p + r
                for cg in scgs:
                    if not cg.converged:
                        cg.step2(r)

                t("other")
                for cg in scgs:
                    msg = cg.check(cp, rsq)
                    if msg:
                        self.log(f"{msg} at iteration {k+1}")
                if sum([cg.converged for cg in scgs]) == ns:
                    return

            self.log(f"NOT converged in {k+1} iterations;  squared residual {cp:e} / {rsq:e}")

        return g.matrix_operator(
            mat=inv,
            vector_space=vector_space,
            accept_guess=(True, False),
            accept_list=lambda src: len(src) * len(self.shifts),
        )
