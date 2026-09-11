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
    def __init__(self, lat, points, code, code_parallel_block_size=None):
        margin = [0] * lat.grid.nd
        for p in points:
            for i in range(lat.grid.nd):
                x = abs(p[i])
                if x > margin[i]:
                    margin[i] = x

        self.padding = g.padded_local_fields(lat, margin)
        self.local_stencil = g.local_stencil.matrix(
            self.padding(lat), points, code, code_parallel_block_size
        )
        self.write_fields = None
        self.verbose_performance = g.default.is_verbose("stencil_performance")

    def data_access_hints(self, write_fields, read_fields, cache_fields):
        self.write_fields = write_fields
        self.read_fields = read_fields
        self.cache_fields = cache_fields

    def __call__(self, *fields):
        fields[0].foundation.stencil.matrix(self, *fields)


def matrix(lat, points, code, code_parallel_block_size=None):
    # check if all points are cartesian
    for p in points:
        if len([s for s in p if s != 0]) > 1:
            return matrix_padded(lat, points, code, code_parallel_block_size)
    return g.local_stencil.matrix(lat, points, code, code_parallel_block_size, local=0)
