#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
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
import sys


class point_manager:
    def __init__(self, point_set):
        self.points = []
        for p in sorted(point_set):
            self.points.append(p)

    def __call__(self, point):
        idx = self.points.index(point)
        assert idx >= 0
        return idx


class parallel_transport_matrix:
    def __init__(self, U, code, n_target):
        self.verbose = g.default.is_verbose("parallel_transport_matrix_performance")

        Nd = len(U)
        Ntarget = n_target
        point_set = set([(0,) * Nd])

        # next parse code for temporaries
        Ntemporary = 0
        for c in code:
            if c[0] >= Ntarget + Ntemporary:
                Ntemporary = c[0] - Ntarget + 1

        # parse code for all paths
        paths = []
        for c in code:
            if isinstance(c[-1], g.path):
                paths.append(c[-1])
            else:
                for f in c[-1]:
                    assert isinstance(f[1], tuple) and len(f[1]) == Nd
                    point_set.add(f[1])

        # save parameters
        self.Ntemporary = Ntemporary
        self.Ntarget = Ntarget
        self.Nd = Nd

        # first get list of all points
        for p in paths:
            coor = [0] * Nd
            for d in p.path:
                if d[1] > 0:
                    for i in range(d[1]):
                        point_set.add(tuple(coor))
                        coor[d[0]] += 1
                else:
                    for i in range(-d[1]):
                        coor[d[0]] -= 1
                        point_set.add(tuple(coor))

        points = point_manager(point_set)

        # list of fields
        _U = list(range(Ntarget + Ntemporary, Ntarget + Ntemporary + Nd))

        # create code for loops
        self.code = []
        for c in code:
            if isinstance(c[-1], g.path):
                coor = [0] * Nd
                factors = []
                for d in c[-1].path:
                    if d[1] > 0:
                        for i in range(d[1]):
                            idx = points(tuple(coor))
                            coor[d[0]] += 1
                            factors.append((_U[d[0]], idx, 0))
                    else:
                        for i in range(-d[1]):
                            coor[d[0]] -= 1
                            idx = points(tuple(coor))
                            factors.append((_U[d[0]], idx, 1))
            else:
                factors = [(f[0], points(f[1]), f[2]) for f in c[-1]]

            self.code.append((c[0], c[1], c[2], factors))

        self.ncode = len(self.code)

        write_fields = list(range(Ntarget))
        read_fields = list(range(Ntarget + Ntemporary, Ntarget + Ntemporary + Nd))

        self.stencil = g.stencil.matrix(U[0], points.points, self.code)
        self.stencil.data_access_hints(write_fields, read_fields, [])

    def __call__(self, U):
        T = [g.lattice(U[0]) for i in range(self.Ntarget)]
        Temp = [g.lattice(U[0]) for i in range(self.Ntemporary)]
        self.stencil(*T, *Temp, *U)

        if self.Ntarget == 1:
            return T[0]

        return T
