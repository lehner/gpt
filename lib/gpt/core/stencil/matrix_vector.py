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


class matrix_vector_padded:
    def __init__(
        self,
        lat_matrix,
        lat_vector,
        points,
        code,
        code_parallel_block_size=None,
        matrix_parity=0,
        vector_parity=0,
    ):
        margin = [0] * lat_matrix.grid.nd
        for p in points:
            for i in range(lat_matrix.grid.nd):
                x = abs(p[i])
                if x > margin[i]:
                    margin[i] = x

        if lat_matrix.grid.cb.n == 1:
            self.matrix_padding = {g.none: g.padded_local_fields(lat_matrix, margin)}
            self.vector_padding = {g.none: g.padded_local_fields(lat_vector, margin)}
        else:
            self.matrix_padding = {}
            self.vector_padding = {}
            m = g.lattice(lat_matrix)
            v = g.lattice(lat_vector)
            for p in [g.even, g.odd]:
                m.checkerboard(p)
                self.matrix_padding[p] = g.padded_local_fields(m, margin)
                v.checkerboard(p)
                self.vector_padding[p] = g.padded_local_fields(v, margin)
        mp = self.matrix_padding[lat_matrix.checkerboard()](lat_matrix)
        vp = self.vector_padding[lat_vector.checkerboard()](lat_vector)
        self.local_stencil = g.local_stencil.matrix_vector(
            mp,
            vp,
            points,
            code,
            code_parallel_block_size,
            local=1,
            matrix_parity=matrix_parity,
            vector_parity=vector_parity,
        )
        self.write_fields = None
        self.verbose_performance = g.default.is_verbose("stencil_performance")
        self.cache_padded_matrix_fields = {}

    def data_access_hints(self, write_fields, read_fields, cache_fields):
        # write and read fields are always vector fields
        # cache fields are always matrix fields
        self.write_fields = write_fields
        self.read_fields = read_fields
        self.cache_fields = cache_fields

    def __call__(self, matrix_fields, vector_fields):

        if self.write_fields is None:
            raise Exception(
                "Generalized matrix_vector stencil needs more information.  Call stencil.data_access_hints."
            )

        t = g.timer("stencil.matrix_vector")
        t("create fields")

        padded_matrix_fields = []
        padded_matrix_field = None
        for i in range(len(matrix_fields)):
            if i in self.cache_padded_matrix_fields:
                padded_matrix_field = self.cache_padded_matrix_fields[i]
            else:
                padded_matrix_field = self.matrix_padding[matrix_fields[i].checkerboard()](
                    matrix_fields[i]
                )
                if i in self.cache_fields:
                    self.cache_padded_matrix_fields[i] = padded_matrix_field
            padded_matrix_fields.append(padded_matrix_field)
        assert padded_matrix_field is not None

        padded_vector_fields = []
        padded_vector_field = None
        for i in range(len(vector_fields)):
            if i in self.read_fields:
                padded_vector_field = self.vector_padding[vector_fields[i].checkerboard()](
                    vector_fields[i]
                )
                padded_vector_fields.append(padded_vector_field)
            else:
                padded_vector_fields.append(None)
        assert padded_vector_field is not None

        for i in range(len(matrix_fields)):
            if padded_matrix_fields[i] is None:
                padded_matrix_fields[i] = g.lattice(padded_matrix_field)

        for i in range(len(vector_fields)):
            if padded_vector_fields[i] is None:
                padded_vector_fields[i] = g.lattice(padded_vector_field)

        t("local stencil")
        self.local_stencil(padded_matrix_fields, padded_vector_fields)
        t("extract")

        for i in self.write_fields:
            self.vector_padding[vector_fields[i].checkerboard()].extract(
                vector_fields[i], padded_vector_fields[i]
            )

        t()

        if self.verbose_performance:
            g.message(t)
        # todo: make use of cache_fields


def matrix_vector(
    lat_matrix,
    lat_vector,
    points,
    code,
    code_parallel_block_size=None,
    matrix_parity=0,
    vector_parity=0,
):
    # check if all points are cartesian
    for p in points:
        if len([s for s in p if s != 0]) > 1:
            return matrix_vector_padded(
                lat_matrix,
                lat_vector,
                points,
                code,
                code_parallel_block_size,
                matrix_parity,
                vector_parity,
            )
    return g.local_stencil.matrix_vector(
        lat_matrix,
        lat_vector,
        points,
        code,
        code_parallel_block_size,
        local=0,
        matrix_parity=matrix_parity,
        vector_parity=vector_parity,
    )
