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


def new_target_list(prototype, n):
    # A fresh target for n outputs of one fused stencil call, at the same
    # reverse-AD nesting depth as `prototype`.  The AD stencil foundation
    # represents the m outputs of a fused call as ONE list node per nesting
    # level (not m sibling nodes), see lib/gpt/ad/reverse/foundation/stencil.py.
    #
    # The depth is read statically: resolving it by evaluating would force a
    # forward pass on a computed prototype and cache a value that a later
    # backward pass then reuses instead of recomputing it from updated leaves.
    r = [g.lattice(prototype.grid, prototype.otype) for i in range(n)]
    for i in range(g.ad.reverse.util.value_depth_static(prototype)):
        r = g.ad.reverse.node(r)
    return r


class parallel_transport_matrix:
    def __init__(self, U, code, n_target):
        self.verbose = g.default.is_verbose("parallel_transport_matrix_performance")

        Nd = len(U)
        Nd_grid = U[0].grid.nd
        Ntarget = n_target
        point_set = set([(0,) * Nd_grid])

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
                    assert isinstance(f[1], tuple) and len(f[1]) == Nd_grid
                    point_set.add(f[1])

        # save parameters
        self.Ntemporary = Ntemporary
        self.Ntarget = Ntarget
        self.Nd = Nd

        # first get list of all points
        for p in paths:
            coor = [0] * Nd_grid
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
                coor = [0] * Nd_grid
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

        # the stencil only needs a prototype lattice (grid/otype); U may be a
        # reverse-AD node, whose grid/otype are those of the value it wraps
        prototype = g.lattice(U[0].grid, U[0].otype)
        self.stencil = g.stencil.matrix(prototype, points.points, self.code)
        self.stencil.data_access_hints(write_fields, read_fields, [])

    def __call__(self, U):
        if isinstance(U[0], g.ad.reverse.node_base) and self.Ntarget > 1:
            # node mode with several outputs: the AD foundation expects them as
            # a single LIST node, not Ntarget sibling nodes.  Hand the caller
            # the elements of that node so the interface matches the plain case.
            # Only g.parallel_transport reaches this (one target per path, no
            # temporaries); a code with temporaries would additionally have to
            # pass them as gradient-free constants.
            assert self.Ntemporary == 0, (
                "parallel_transport_matrix: node mode with several outputs does "
                "not support temporaries")
            T = new_target_list(U[0], self.Ntarget)
            self.stencil(T, *U)
            return [T[i] for i in range(self.Ntarget)]

        # x.new() allocates a fresh object of the same type (grid/otype), for
        # both plain lattices and reverse-AD nodes (the latter resolves the
        # stencil call to the AD foundation); the stencil overwrites the
        # targets, so the initial contents are irrelevant
        T = [U[0].new() for i in range(self.Ntarget)]
        Temp = [U[0].new() for i in range(self.Ntemporary)]
        self.stencil(*T, *Temp, *U)

        if self.Ntarget == 1:
            return T[0]

        return T
