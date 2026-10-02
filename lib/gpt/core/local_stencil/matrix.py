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
import gpt as g
from gpt.core import auto_tuned_class, auto_tuned_method
from gpt.core.local_stencil.cse import cse as _cse
import hashlib


def hash_code(code):
    return str(len(code)) + "-" + str(hashlib.sha256(str(code).encode("utf-8")).hexdigest())


def parse(c):
    if isinstance(c, tuple):
        assert len(c) == 4
        return {"target": c[0], "accumulate": c[1], "weight": c[2], "factor": c[3]}
    return c


class matrix(auto_tuned_class):
    # comm_type: 0 = cartesian stencil (points on the axes, Grid's halo
    # exchange), 1 = no communication (the shifts wrap around within the
    # local fields, e.g. for halo-padded fields), 2 = general stencil (any
    # points, halo exchange of cgpt's foundation layer)
    #
    # temporaries: field indices used as per-site temporaries (read and
    # written only at the zero shift).  They are owned by the stencil (a
    # buffer of one block of osites_per_cache_block outer sites each,
    # allocated once; 0: default block size) and NOT passed by the caller,
    # who passes the remaining fields in index order.
    #
    # cse: combine common subexpressions (see local_stencil/cse.py) in the
    # executed kernel (True, or the minimal number of uses of a temporary).
    # The stencil keeps its original points/code/temporaries (they define its
    # meaning, e.g. for the AD derivation); the executed plan is `executed`.
    def __init__(
        self,
        lat,
        points,
        code,
        code_parallel_block_size=None,
        comm_type=1,
        temporaries=(),
        osites_per_cache_block=0,
        cse=False,
    ):
        self.points = points
        self.code = [parse(c) for c in code]
        self.temporaries = tuple(sorted(temporaries))
        self.code_parallel_block_size = code_parallel_block_size
        # the executed plan (temporaries need a kernel without communication
        # in the kernel, comm_type 1 or 2, and a single code-parallel block)
        self.executed = None
        if cse and comm_type != 0 and code_parallel_block_size in (None, len(code)):
            self.executed = _cse(points, self.code, self.temporaries, 2 if cse is True else cse)
        if self.executed is not None:
            points, code, temporaries = self.executed
            code = [parse(c) for c in code]
        else:
            code = self.code
        if code_parallel_block_size is None:
            code_parallel_block_size = len(code)
        self.obj = cgpt.stencil_matrix_create(
            lat.v_obj[0],
            lat.grid.obj,
            points,
            code,
            code_parallel_block_size,
            comm_type,
            list(temporaries),
            osites_per_cache_block,
        )

        # auto tuner: (fast_osites, threads per block of accelerator_for, 0 =
        # the current value, i.e., --accelerator-threads)
        tag = f"local_matrix({lat.otype.__name__}, {lat.grid.describe()}, {str(points)}, {code_parallel_block_size}, {hash_code(code)}, {comm_type}, {sorted(temporaries)}, {osites_per_cache_block}, threads)"
        super().__init__(
            tag,
            [(fast_osites, threads) for fast_osites in [0, 1] for threads in g.default.auto_tune_threads],
            (0, 0),
        )

    @auto_tuned_method()
    def _exec(self, params, *fields):
        fast_osites, threads = params
        with g.accelerator.threads(threads):
            cgpt.stencil_matrix_execute(self.obj, list(fields), fast_osites)

    def __call__(self, *fields):
        return fields[0].foundation.local_stencil.matrix(self, *fields)

    def __del__(self):
        cgpt.stencil_matrix_delete(self.obj)

    def data_access_hints(self, *hints):
        pass
