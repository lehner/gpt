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
import cgpt
from gpt.core import auto_tuned_class, auto_tuned_method
import hashlib


def hash_code(code):
    return str(len(code)) + "-" + str(hashlib.sha256(str(code).encode("utf-8")).hexdigest())


def parse(c):
    if isinstance(c, tuple):
        assert len(c) == 4
        return {"target": c[0], "accumulate": c[1], "weight": c[2], "factor": c[3]}
    return c


class matrix(auto_tuned_class):
    def __init__(self, lat, points, code, code_parallel_block_size=None, local=1):
        self.points = points
        self.code = [parse(c) for c in code]
        self.code_parallel_block_size = code_parallel_block_size
        if code_parallel_block_size is None:
            code_parallel_block_size = len(code)
        self.obj = cgpt.stencil_matrix_create(
            lat.v_obj[0], lat.grid.obj, points, self.code, code_parallel_block_size, local
        )

        # auto tuner
        tag = f"local_matrix({lat.otype.__name__}, {lat.grid.describe()}, {str(points)}, {code_parallel_block_size}, {hash_code(code)}, {local})"
        super().__init__(tag, [0, 1], 0)

    @auto_tuned_method
    def __call__(self, fast_osites, *fields):
        cgpt.stencil_matrix_execute(self.obj, list(fields), fast_osites)

    def __del__(self):
        cgpt.stencil_matrix_delete(self.obj)

    def data_access_hints(self, *hints):
        pass
