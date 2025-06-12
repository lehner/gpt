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
import gpt as g
import cgpt
import numpy as np


class blas:
    def __init__(self):
        self.obj = cgpt.create_blas()
        self.references = []

    def __del__(self):
        cgpt.delete_blas(self.obj)

    def accumulate(self, buffers):
        assert all([ buffers[0].shape == b.shape for b in buffers ])
        cgpt.blas_accumulate(
            self.obj,
            int(np.prod(buffers[0].shape)),
            [x.view for x in buffers],
            buffers[0].dtype
        )

    def gemm(self, alpha, bv_A, bv_B, beta, bv_C, precision=None):
        bv_A = g.util.to_list(bv_A)
        bv_B = g.util.to_list(bv_B)
        bv_C = g.util.to_list(bv_C)
        assert len(bv_A) == len(bv_B) and len(bv_A) == len(bv_C)
        
        # add references so that memory used will not be deallocated
        self.references.append([x.buffer.view for x in bv_A + bv_B + bv_C])

        assert all([ x.buffer.dtype == bv_A[0].buffer.dtype for x in bv_A + bv_B + bv_C ])

        op_A = bv_A[0].op
        op_B = bv_B[0].op
        op_C = bv_C[0].op

        assert all([ x.op == op_A for x in bv_A ])
        assert all([ x.op == op_B for x in bv_B ])
        assert all([ x.op == op_C for x in bv_C ])

        # AB = C row-major is BA = C column-major
        # numpy and usual Grid order is row-major
        # GridBlas order is column-major
        bv_A, bv_B, op_A, op_B = bv_B, bv_A, op_B, op_A

        # operators
        N = bv_C[0].op_N
        T = bv_C[0].op_T
        C = bv_C[0].op_C

        # re-distribute operators away from C
        if op_C & C:
            op_C ^= C
            op_A ^= C
            for x in bv_B:
                x.op ^= C

        if op_C & T:
            op_C ^= T
            op_A ^= T
            op_B ^= T
            bv_A, bv_B, op_A, op_B = bv_B, bv_A, op_B, op_A

        # none shall remain
        assert op_C == N

        # for now do not allow just complex conjugation (blas does not support this)
        assert op_A != C
        assert op_B != C
        assert op_C != C

        # m, n, k, Amk, Bkn, Cmn
        a, b = bv_A[0].buffer.shape[-2:]
        c, d = bv_B[0].buffer.shape[-2:]
        e, f = bv_C[0].buffer.shape[-2:]

        assert all([ x.buffer.shape == bv_A[0].buffer.shape for x in bv_A ])
        assert all([ x.buffer.shape == bv_B[0].buffer.shape for x in bv_B ])
        assert all([ x.buffer.shape == bv_C[0].buffer.shape for x in bv_C ])

        j, i = (a, b) if not (op_A & T) else (b, a)
        k, _j = (c, d) if not (op_B & T) else (d, c)
        _k, _i = e, f

        assert j == _j
        assert i == _i
        assert k == _k

        if precision is None:
            precision = "default"
            
        cgpt.blas_gemm(
            self.obj,
            i,
            k,
            j,
            alpha,
            [x.buffer.view for x in bv_A],
            [np.ascontiguousarray(x.idx, dtype=np.int64) for x in bv_A],
            op_A,
            [x.buffer.view for x in bv_B],
            [np.ascontiguousarray(x.idx, dtype=np.int64) for x in bv_B],
            op_B,
            beta,
            [x.buffer.view for x in bv_C],
            [np.ascontiguousarray(x.idx, dtype=np.int64) for x in bv_C],
            bv_C[0].buffer.dtype,
            precision
        )
        return self

    def inv(self, bv_A, bv_C):
        # add references so that memory used will not be deallocated
        self.references.append(bv_A.buffer.view)
        self.references.append(bv_C.buffer.view)

        assert bv_A.buffer.dtype == bv_C.buffer.dtype

        # cannot work with transformations
        assert bv_A.op == bv_C.op_N
        assert bv_C.op == bv_C.op_N

        # dimension
        n1, n2 = bv_A.buffer.shape[-2:]
        n3, n4 = bv_C.buffer.shape[-2:]

        assert n1 == n2 and n1 == n3 and n1 == n4

        cgpt.blas_inv(
            self.obj,
            n1,
            bv_A.buffer.view,
            np.ascontiguousarray(bv_A.idx, dtype=np.int64),
            bv_C.buffer.view,
            np.ascontiguousarray(bv_C.idx, dtype=np.int64),
            bv_C.buffer.dtype,
        )
        return self

    def det(self, bv_A, bv_C):
        # add references so that memory used will not be deallocated
        self.references.append(bv_A.buffer.view)
        self.references.append(bv_C.buffer.view)

        assert bv_A.buffer.dtype == bv_C.buffer.dtype

        # cannot work with transformations
        assert bv_A.op == bv_C.op_N
        assert bv_C.op == bv_C.op_N

        # dimension
        n1, n2 = bv_A.buffer.shape[-2:]

        assert n1 == n2

        cgpt.blas_det(
            self.obj,
            n1,
            bv_A.buffer.view,
            np.ascontiguousarray(bv_A.idx, dtype=np.int64),
            bv_C.buffer.view,
            np.ascontiguousarray(bv_C.idx, dtype=np.int64),
            bv_C.buffer.dtype,
        )
        return self

    def __call__(self):
        cgpt.blas_execute(self.obj)
        return self
