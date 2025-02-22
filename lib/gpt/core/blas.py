#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import numpy as np


class blas:
    def __init__(self):
        self.obj = cgpt.create_blas()
        self.references = []

    def __del__(self):
        cgpt.delete_blas(self.obj)

    def gemm(self, alpha, bv_A, bv_B, beta, bv_C):
        # add references so that memory used will not be deallocated
        self.references.append(bv_A.buffer.view)
        self.references.append(bv_B.buffer.view)
        self.references.append(bv_C.buffer.view)

        assert bv_A.buffer.dtype == bv_B.buffer.dtype
        assert bv_A.buffer.dtype == bv_C.buffer.dtype

        # re-distribute operators away from C
        if bv_C.op & bv_C.op_C:
            bv_C.op ^= bv_C.op_C
            bv_A.op ^= bv_C.op_C
            bv_B.op ^= bv_C.op_C

        if bv_C.op & bv_C.op_T:
            bv_C.op ^= bv_C.op_T
            bv_A.op ^= bv_C.op_T
            bv_B.op ^= bv_C.op_T
            bv_A, bv_B = bv_B, bv_A

        # none shall remain
        assert bv_C.op == bv_C.op_N

        # m, n, k, Amk, Bkn, Cmn
        if bv_A.op & bv_A.op_T:
            k, m = bv_A.buffer.shape[-2:]
        else:
            m, k = bv_A.buffer.shape[-2:]

        if bv_B.op & bv_A.op_T:
            _k, n = bv_B.buffer.shape[-2:]
        else:
            n, _k = bv_B.buffer.shape[-2:]

        _n, _m = bv_C.buffer.shape[-2:]

        assert m == _m
        assert n == _n
        assert k == _k

        cgpt.blas_gemm(
            self.obj,
            m,
            n,
            k,
            alpha,
            bv_A.buffer.view,
            np.ascontiguousarray(bv_A.idx, dtype=np.int64),
            bv_A.op,
            bv_B.buffer.view,
            np.ascontiguousarray(bv_B.idx, dtype=np.int64),
            bv_B.op,
            beta,
            bv_C.buffer.view,
            np.ascontiguousarray(bv_C.idx, dtype=np.int64),
            bv_C.buffer.dtype,
        )

    def __call__(self):
        cgpt.blas_execute(self.obj)
