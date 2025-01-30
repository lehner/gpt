#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2024  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


class algebra_laplace:
    def __init__(self, U, mass=0.0):
        self.U = U
        self.grid = U[0].grid
        self.nd = len(U)
        assert self.nd == self.grid.nd

        # vectors
        origin = tuple([0 for j in range(self.nd)])
        evec = [tuple([1 if i == j else 0 for j in range(self.nd)]) for i in range(self.nd)]
        nevec = [tuple([-x for x in y]) for y in evec]

        _P = 0
        _Sp = list(range(1, 1 + self.nd))
        _Sm = list(range(1 + self.nd, 1 + 2 * self.nd))

        # indices
        _dU = list(range(self.nd))
        _dV = [x + self.nd for x in _dU]
        _sU = [x + self.nd for x in _dV]
        _sV = [x + self.nd for x in _sU]

        # code
        code = []
        for nu in range(self.nd):
            code.append((_dU[nu], -1, 1.0, [(_sU[nu], _P, 0)]))
            code.append((_dV[nu], -1, -2 * self.nd / 16 + mass, [(_sV[nu], _P, 0)]))

            for mu in range(self.nd):
                code.append(
                    (
                        _dV[nu],
                        _dV[nu],
                        1 / 16,
                        [(_sU[mu], _P, 0), (_sV[nu], _Sp[mu], 0), (_sU[mu], _P, 1)],
                    )
                )

                code.append(
                    (
                        _dV[nu],
                        _dV[nu],
                        1 / 16,
                        [(_sU[mu], _Sm[mu], 1), (_sV[nu], _Sm[mu], 0), (_sU[mu], _Sm[mu], 0)],
                    )
                )

        # stencil
        self.st = g.stencil.matrix(U[0], [origin] + evec + nevec, code)

        # indices for gradient (ret, left, U, right)
        _ret = list(range(self.nd))
        _left = [x + self.nd for x in _ret]
        _U = [x + self.nd for x in _left]
        _right = [x + self.nd for x in _U]

        # code
        code = []
        for mu in range(self.nd):
            for nu in range(self.nd):
                code.append(
                    (
                        _ret[mu],
                        _ret[mu] if nu > 0 else -1,
                        1j / 32,
                        [
                            (_U[mu], _P, 0),
                            (_right[nu], _Sp[mu], 0),
                            (_U[mu], _P, 1),
                            (_left[nu], _P, 1),
                        ],
                    )
                )
                code.append(
                    (
                        _ret[mu],
                        _ret[mu],
                        -1j / 32,
                        [
                            (_right[nu], _P, 0),
                            (_U[mu], _P, 0),
                            (_left[nu], _Sp[mu], 1),
                            (_U[mu], _P, 1),
                        ],
                    )
                )
                code.append(
                    (
                        _ret[mu],
                        _ret[mu],
                        -1j / 32,
                        [
                            (_left[nu], _P, 1),
                            (_U[mu], _P, 0),
                            (_right[nu], _Sp[mu], 0),
                            (_U[mu], _P, 1),
                        ],
                    )
                )
                code.append(
                    (
                        _ret[mu],
                        _ret[mu],
                        1j / 32,
                        [
                            (_U[mu], _P, 0),
                            (_left[nu], _Sp[mu], 1),
                            (_U[mu], _P, 1),
                            (_right[nu], _P, 0),
                        ],
                    )
                )

        # stencil
        self.st_pg = g.stencil.matrix(U[0], [origin] + evec + nevec, code)

    def __call__(self, dst, src):
        # src = U + V
        assert len(dst) == 2 * self.nd
        assert len(src) == 2 * self.nd

        for i in range(len(dst)):
            dst[i].otype = src[i].otype

        self.st(*dst, *src)

    def projected_gradient(self, left, U, right):
        assert len(left) == self.nd
        assert len(right) == self.nd
        assert len(U) == self.nd

        ret = [g.group.cartesian(u) for u in U]
        self.st_pg(*ret, *left, *U, *right)
        for mu in range(self.nd):
            # could create a stencil version of project.traceless_hermitian that works on all ret[mu] at the same time
            ret[mu] @= g.qcd.gauge.project.traceless_hermitian(ret[mu])
        return ret

    def inverse(self, inverter, function=None):

        if function is None:
            function = self

        def _mat(dst, src):
            U = src[0 : self.nd]

            def _inv_mat(_dst, _src):
                function(U + _dst, U + _src)

            inv_mat = g.matrix_operator(mat=_inv_mat, accept_list=True, accept_guess=(True, False))

            im = inverter(inv_mat)

            g.eval(dst[self.nd :], im * src[self.nd :])

            for d in dst[self.nd :]:
                d.otype = src[self.nd].otype

            g.copy(dst[0 : self.nd], U)

        return g.matrix_operator(mat=_mat, accept_list=True)


class algebra_laplace_polynomial:
    def __init__(self, U, scale, mass):
        self.scale = scale
        self.lap = [g.qcd.gauge.algebra_laplace(U, mass=m) for m in mass]
        self.mass = mass
        self.nd = len(U)

    def __call__(self, dst, src):
        for lap in self.lap:
            lap(dst, src)
            src = g.copy(dst)
        g(dst[self.nd :], self.scale * g.expr(src[self.nd :]))

    def inverse(self, solver):
        return self.lap[0].inverse(solver, self)

    def projected_gradient(self, left, U, right):
        ret = None
        right_i = [right]
        left_i = [left]
        for i in range(len(self.mass) - 1):
            right_i.append(g.copy(right_i[-1]))
            self.lap[len(self.mass) - 1 - i](U + right_i[-1], U + right_i[-2])

            left_i.append(g.copy(left_i[-1]))
            self.lap[i](U + left_i[-1], U + left_i[-2])

        for i in range(len(self.mass)):
            ret_i = self.lap[i].projected_gradient(left_i[i], U, right_i[len(self.mass) - 1 - i])
            if ret is None:
                ret = [g(self.scale * x) for x in ret_i]
            else:
                for j in range(len(ret)):
                    ret[j] += self.scale * ret_i[j]
        return ret
