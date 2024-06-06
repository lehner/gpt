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
    def __init__(self, U):
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
        _dst = list(range(self.nd))
        _src = [x + self.nd for x in _dst]
        _U = [x + self.nd for x in _src]

        # code
        code = []
        for nu in range(self.nd):
            for mu in range(self.nd):
                code.append(
                    (
                        _dst[nu],
                        -1 if mu == 0 else _dst[nu],
                        1 / 16,
                        [(_U[mu], _P, 0), (_src[nu], _Sp[mu], 0), (_U[mu], _P, 1)],
                    )
                )

                code.append(
                    (
                        _dst[nu],
                        _dst[nu],
                        1 / 16,
                        [(_U[mu], _Sm[mu], 1), (_src[nu], _Sm[mu], 0), (_U[mu], _Sm[mu], 0)],
                    )
                )

                code.append((_dst[nu], _dst[nu], -2 / 16, [(_src[nu], _P, 0)]))

        # stencil
        self.st = g.stencil.matrix(U[0], [origin] + evec + nevec, code)

    def __call__(self, dst, src):
        assert len(src) == len(dst)
        assert len(src) == self.nd

        self.st(*dst, *src, *self.U)
