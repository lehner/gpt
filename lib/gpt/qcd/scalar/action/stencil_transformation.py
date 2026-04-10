#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


class stencil_transformation:
    def __init__(self, U, description):
        self.description = description
        points = len(description)
        vecs = [x[1] for x in description]
        weights = [x[0] for x in description]
        assert points > 0
        self.nd = len(vecs[0])

        # make sure first element is zero-shift
        assert vecs[0] == tuple(0 for mu in range(self.nd))
        
        # indices
        _dU = list(range(self.nd))
        _dV = [x + self.nd for x in _dU]
        _sU = [x + self.nd for x in _dV]
        _sV = [x + self.nd for x in _sU]
        _P = 0

        code = []
        for nu in range(self.nd):
            code.append((_dU[nu], -1, 1.0, [(_sU[nu], _P, 0)]))
            for i in range(points):
                code.append((_dV[nu], -1 if i == 0 else _dV[nu], weights[i], [(_sV[nu], i, 0)]))

        # stencil
        self.st = g.stencil.matrix(U[0], vecs, code)
        self.st.data_access_hints(_dU + _dV, _sU + _sV, [])

    def inverse(self, inverter):
        return g.qcd.gauge.algebra_laplace.inverse(self, inverter)

    def __call__(self, dst, src):
        self.st(*dst, *src)
        for x, y in zip(dst, src):
            x.otype = y.otype

    def projected_gradient(self, left, U, right):
        r = [g.group.cartesian(u) for u in U]
        for x in r:
            x[:] = 0
        assert False
        return r
