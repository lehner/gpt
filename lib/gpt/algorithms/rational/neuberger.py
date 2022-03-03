#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2022  Raphael Lehner (raphael.lehner@physik.uni-regensburg.de)
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
from gpt.algorithms.inverter import multi_shift_fgmres, multi_shift_fom

#
# neuberger_inverse_square_root approximates 1/sqrt(x^2) with
#         (z+u_1) (z+u_2)
#  norm * ------- ------- ...
#         (z+v_1) (z+v_2)
#
class neuberger_inverse_square_root:
    def __init__(self, m, r, order):
        assert 0.0 < r and r < m
        self.m = m
        self.r = r
        self.n = order

        c = 1.0 / ((m + r) * (m - r)) ** 0.5
        d = ((m + r) / (m - r)) ** 0.5
        e = np.pi / 2.0 / order
        A = np.sum(
            [1.0 / np.cos(e * (i + 0.5)) ** 2.0 for i in range(order)]
        )
        a = np.array(
            [-(np.tan(e / 2.0 * i) / c) ** 2.0 for i in range(1, 2 * order)]
        )

        self.zeros = a[1::2]
        self.poles = a[0::2]
        self.norm = A / c / order
        self.delta = 2.0 / (((d + 1.0) / ((d - 1.0))) ** (2.0 * order) - 1.0)

    def __str__(self):
        out = f"Neuberger approx of 1/sqrt(x^2) with {self.n} poles "
        out += f"in the circle C(m = {self.m}, r = {self.r})\n"
        out += f"   relative error delta = {self.delta:e}"
        return out


class neuberger_sign(base_iterative):
    @g.params_convention(eps=1e-12, m=None, r=None, inverter=None)
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.m = params["m"]
        self.r = params["r"]
        self.inverter = params["inverter"]

    def order(self, eps):
        d = ((self.m + self.r) / (self.m - self.r)) ** 0.5
        s = np.log(eps / (eps + 2.0)) / 2.0 / np.log((d - 1.0) / (d + 1.0))
        return int(np.ceil(s))

    def __call__(
        self,
        mat,
        msq_evals=None,
        msq_evec=None,
        msq_levec=None,
        msq_revec=None,
    ):

        vector_space = None
        if type(mat) == g.matrix_operator:
            vector_space = mat.vector_space
            mat = mat.mat
            # remove wrapper for performance benefits

        # evals and evec for LR deflation
        evals = msq_evals
        levec = msq_evec if msq_evec is not None else msq_levec
        revec = msq_evec if msq_evec is not None else msq_revec
        if evals is not None:
            assert len(evals) == len(levec)
            assert len(evals) == len(revec)

        # rational function
        rf_eps = self.eps / (2.0 + self.eps)
        order = self.order(rf_eps)
        neu = neuberger_inverse_square_root(self.m, self.r, order)
        rf = g.algorithms.rational.rational_function(
            neu.zeros, neu.poles, neu.norm
        )

        # multishift inverter
        ms_eps = self.eps / 2.0
        self.inverter.eps = ms_eps
        self.inverter.shifts = -1.0 * rf.poles

        @self.timed_function
        def sign(dst, src, t):

            # timing
            t("setup")

            # fields
            dst[:] = 0
            deflated_src, mmp = g.copy(src), g.copy(src)
            psis = [g.copy(src) for i in range(len(rf.poles))]

            # squared matrix
            def msq(dst, src):
                mat(mmp, src)
                mat(dst, mmp)

            # inverters with projection step
            ps = (multi_shift_fom, multi_shift_fgmres)

            # setup multishift inverter
            if isinstance(self.inverter, ps) and evals is not None:

                # projection matrix for LR deflation
                def P(dst, src):
                    dst @= g.copy(src)
                    for l, r in zip(levec, revec):
                        v = g.inner_product(l, src)
                        dst -= r * v
                ms_inv = self.inverter(msq, P)
            else:
                ms_inv = self.inverter(msq)

            # LR deflation
            if evals is not None:

                t("deflation")
                for e, l, r in zip(evals, levec, revec):
                    v = g.inner_product(l, deflated_src)
                    dst += r * v / e ** 0.5
                    deflated_src -= r * v

            t("multi_shift")
            ms_inv(psis, deflated_src)

            t("linear_combination")
            for psi, r in zip(psis, rf.r):
                dst += psi * r * rf.norm

            t("matrix")
            mat(mmp, dst)
            dst @= mmp

        return g.matrix_operator(
            mat=sign, inv_mat=sign, accept_guess=(True, False), vector_space=vector_space
        )
