#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


class matrix_padded:
    # temporaries: per-site temporaries owned by the kernel (see
    # g.local_stencil.matrix); they are not passed by the caller, whose fields
    # (and data access hints) are the remaining ones in index order
    def __init__(self, lat, points, code, code_parallel_block_size=None, temporaries=()):
        margin = [0] * lat.grid.nd
        for p in points:
            for i in range(lat.grid.nd):
                x = abs(p[i])
                if x > margin[i]:
                    margin[i] = x

        self.padding = g.padded_local_fields(lat, margin)
        self.local_stencil = g.local_stencil.matrix(
            self.padding(lat), points, code, code_parallel_block_size, temporaries=temporaries
        )
        temps = self.local_stencil.temporaries

        def position(i):
            # position of code field i among the fields the caller passes
            return i - sum(1 for t in temps if t < i)

        # from the code: the referenced fields, and the targets whose first
        # write is fresh (accumulate = -1) and not preceded by a read of them;
        # their previous content is never used
        first, read_before = {}, set()
        self.referenced = set()
        for c in self.local_stencil.code:
            for f in c["factor"]:
                if f[0] not in first:
                    read_before.add(f[0])
                self.referenced.add(f[0])
            first.setdefault(c["target"], c["accumulate"])
            self.referenced.add(c["target"])
            if c["accumulate"] != -1:
                self.referenced.add(c["accumulate"])
        self.fresh_targets = {
            position(t) for t, a in first.items() if a == -1 and t not in read_before and t not in temps
        }
        self.referenced = {position(i) for i in self.referenced if i not in temps}
        self.write_fields = None
        self.verbose_performance = g.default.is_verbose("stencil_performance")

    def data_access_hints(self, write_fields, read_fields, cache_fields):
        self.write_fields = write_fields
        self.read_fields = read_fields
        self.cache_fields = cache_fields

    def __call__(self, *fields, padded=None):
        # padded: optional {field index: padded copy} of READ-ONLY fields that
        # the caller already holds in this stencil's padding domain (e.g. the
        # padded inputs of an earlier stencil run on the same, unchanged
        # fields); they are used as they are instead of being copied again.
        # Returns the list of padded fields the kernel ran on (for the plain
        # lattice foundation), so a caller may keep read-only padded inputs.
        if padded is None:
            return fields[0].foundation.stencil.matrix(self, *fields)
        return fields[0].foundation.stencil.matrix(self, *fields, padded=padded)


def matrix(lat, points, code, code_parallel_block_size=None, temporaries=()):
    # kernel-owned temporaries need the local (padded) kernel
    if len(temporaries) > 0:
        return matrix_padded(lat, points, code, code_parallel_block_size, temporaries)
    # check if all points are cartesian
    for p in points:
        if len([s for s in p if s != 0]) > 1:
            return matrix_padded(lat, points, code, code_parallel_block_size)
    return g.local_stencil.matrix(lat, points, code, code_parallel_block_size, local=0)
