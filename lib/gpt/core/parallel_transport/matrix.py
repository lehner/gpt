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


def parallel_transport_weighted(links, entries, keys=None):
    # entries: [(weight, path), ...] -- one weighted transported product each,
    # as the gauge smears describe a staple sum.
    #
    # Entries that SHARE a weight are accumulated into a single stencil target,
    # so N paths cost one output field instead of N, and the weighted sum the
    # caller would otherwise build from N node multiplies and N-1 node adds
    # collapses to (at most) one multiply per distinct weight.  That removes
    # those ops from the primal graph AND from the several-times-larger
    # derivative graph a reverse pass builds out of it.
    #
    # A numeric weight is folded into the kernel's own coefficient and costs
    # nothing at all.  A field- or node-valued weight cannot be: it is applied
    # once to the accumulated group instead.  Note it must NOT be folded in as
    # an extra stencil FACTOR -- the adjoint of a k-factor entry is k entries
    # of k factors, so one more factor grows the adjoint's arithmetic
    # quadratically (for these 4-link paths, 24 entries/72 multiplies would
    # become 30/120, and one level deeper 288 would become ~600).
    #
    # `keys` optionally tags each entry; entries merge only when their key AND
    # their weight agree.  A smear whose description spans several directions
    # in one stencil passes the direction index, so directions stay separate.
    #
    # Returns (transport, group_info): transport(links) yields one field per
    # group in group order, and group_info[k] is (key, post_weight) for group
    # k, where post_weight is None when the weight is already folded into the
    # kernel and otherwise the weight to apply to that group.
    if keys is None:
        keys = [None] * len(entries)
    assert len(keys) == len(entries)

    groups = []
    group_info = []
    index = {}
    for (weight, path), key in zip(entries, keys):
        numeric = g.util.is_num(weight)
        ident = (key, "numeric" if numeric else id(weight))
        if ident not in index:
            index[ident] = len(groups)
            groups.append([])
            group_info.append((key, None if numeric else weight))
        groups[index[ident]].append((complex(weight) if numeric else 1.0, path))

    code = []
    for target, group in enumerate(groups):
        for k, (w, path) in enumerate(group):
            # the first write of a target is fresh; the rest accumulate into it
            code.append((target, -1 if k == 0 else target, w, path))

    ptm = parallel_transport_matrix(links, code, len(groups))

    def transport(links):
        return g.util.to_list(ptm(links))

    return transport, group_info
