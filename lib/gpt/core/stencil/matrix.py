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


# TODO: explore this thoroughly
# - SIMD mask in lat.grid restrictions?  dimensions plus margin needs to play nice with simd.
# - overlap comms and compute; need padding.start_communicate and padding.wait_communicate
#   and a list of margin and inner points
# - SIMD in multi-rhs ?  maybe add --simd_mask flag to command line ?
# - should do margins automatically seems best
class matrix:
    def __init__(self, lat, points, write_fields, read_fields, code, code_parallel_block_size=None):
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
        self.write_fields = write_fields
        self.read_fields = read_fields
        self.verbose_performance = g.default.is_verbose("stencil_performance")

    def __call__(self, *fields):
        if self.verbose_performance:
            t = g.timer("stencil.matrix")
            t("create fields")
        padded_fields = []
        padded_field = None
        for i in range(len(fields)):
            if i in self.read_fields:
                padded_field = self.padding(fields[i])
                padded_fields.append(padded_field)
            else:
                padded_fields.append(None)
        assert padded_field is not None
        for i in range(len(fields)):
            if padded_fields[i] is None:
                padded_fields[i] = g.lattice(padded_field)
        if self.verbose_performance:
            t("local stencil")
        self.local_stencil(*padded_fields)
        if self.verbose_performance:
            t("extract")
        for i in self.write_fields:
            self.padding.extract(fields[i], padded_fields[i])
        if self.verbose_performance:
            t()
            g.message(t)
