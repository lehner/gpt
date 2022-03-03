#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2022  Mattia Bruno
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

# zolotarev_inverse_square_root approximates 1/sqrt(x^2) with
#         (z+u_1) (z+u_2)
#  norm * ------- ------- ...
#         (z+v_1) (z+v_2)
#
# for more details see documentation/algorithms/rational.ipynb

#
# ellipj code inspired by http://www.netlib.org/cephes/
#
def ellipj(u, m):
    EPS = 1e-16
    a = np.zeros((9,), dtype=np.float64)
    c = np.zeros((9,), dtype=np.float64)

    a[0] = 1.0
    b = np.sqrt(1.0 - m)
    c[0] = np.sqrt(m)
    twon = 1.0
    i = 0

    while abs(c[i] / a[i]) > EPS:
        if i > 7:
            print("Warning ellipj overflow")
            break
        ai = a[i]
        i += 1
        c[i] = (ai - b) / 2.0
        t = np.sqrt(ai * b)
        a[i] = (ai + b) / 2.0
        b = t
        twon *= 2.0

    phi = twon * a[i] * u
    K = np.pi / (2.0 * a[i])
    while i > 0:
        t = c[i] * np.sin(phi) / a[i]
        b = phi
        phi = (np.arcsin(t) + phi) / 2.0
        i -= 1

    t = np.sin(phi)
    sn = t
    cn = np.cos(phi)
    dn = np.sqrt(1.0 - m * t * t)
    return [sn, cn, dn, K]


def zolotarev_approx_inverse_square_root(n, eps):
    a = np.zeros((2 * n,))
    c = np.zeros((2 * n,))

    k = np.sqrt(1 - eps)
    _, _, _, Kk = ellipj(0, k)

    v = Kk / (2 * n + 1)
    for i in range(2 * n):
        sn, cn, dn, _ = ellipj((i + 1) * v, k)
        a[i] = (cn / sn) ** 2
        c[i] = sn ** 2
    # index go from 1 to 2*n
    c_odd = np.prod(c[0::2])
    c_even = np.prod(c[1::2])

    d = np.power(k, 2 * n + 1) * c_odd ** 2
    den = 1 + np.sqrt(1 - d * d)
    A = 2.0 / den * c_odd / c_even

    delta = d ** 2 / den ** 2
    return [A, a[0::2], a[1::2], delta]


# approximate g(x) = 1/sqrt(x^2) in the range ra < x < rb,
# with A \prod_i (x*x - u_i) / (x*x - v_i)
class zolotarev_inverse_square_root:
    def __init__(self, low, high, order):
        self.ra = low
        self.rb = high
        self.n = order

        eps = (self.ra / self.rb) ** 2
        A, u, v, self.delta = zolotarev_approx_inverse_square_root(self.n, eps)

        self.zeros = -u * self.rb ** 2
        self.poles = -v * self.rb ** 2
        self.norm = A / self.rb

    def __str__(self):
        out = f"Zolotarev approx of 1/sqrt(x^2) with {self.n} poles in the range [{self.ra},{self.rb}]\n"
        out += f"   relative error delta = {self.delta}"
        return out


class zolotarev_sign(base_iterative):
    @g.params_convention(eps=1e-12, low=None, high=None, inverter=None)
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.low = params["low"]
        self.high = params["high"]
        self.inverter = params["inverter"]

    # approximation from arXiv:hep-lat/0206007v2
    def order(self, eps):
        b = (self.high / self.low) ** 2.0
        A = 0.465 * np.log(b) ** 0.596
        c = 9.17 * np.log(b) ** -0.774
        s = np.log(2.0 * A / eps) / c
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
        zol = zolotarev_inverse_square_root(self.low, self.high, order)
        rf = g.algorithms.rational.rational_function(
            zol.zeros[0:-1], zol.poles, zol.norm
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
            deflated_src = g.copy(src)
            mms, mmp = g.copy(dst), g.copy(dst)
            psis = [g.copy(src) for i in range(len(rf.poles))]

            # squared matrix
            def msq(dst, src):
                mat(mms, src)
                mat(dst, mms)

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
                mmp += psi * r * rf.norm

            t("matrix")
            msq(deflated_src, mmp)
            mmp @= deflated_src - mmp * zol.zeros[-1] + dst
            mat(dst, mmp)

        return g.matrix_operator(
            mat=sign, inv_mat=sign, accept_guess=(True, False), vector_space=vector_space
        )
