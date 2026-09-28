#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2024  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt.core.foundation.lattice.matrix.exp
import gpt, cgpt
import numpy as np


def _as_matrices(buf, shape, sites):
    # packed per-site tensor -> (sites, n, n); a tensor (n1, n1, n2, n2) is
    # the matrix with the index pair (i1 i2) (row) and (j1 j2) (column)
    if len(shape) == 2:
        return buf.reshape((sites,) + tuple(shape))
    n1, n2 = shape[0], shape[2]
    buf = buf.reshape((sites, n1, n1, n2, n2)).transpose(0, 1, 3, 2, 4)
    return buf.reshape((sites, n1 * n2, n1 * n2))


def _from_matrices(buf, shape, sites):
    if len(shape) == 2:
        return buf
    n1, n2 = shape[0], shape[2]
    return buf.reshape((sites, n1, n2, n1, n2)).transpose(0, 1, 3, 2, 4)


def _batched(A, target, job):
    # Per-site inverse / determinant as one batched BLAS call (cuBLAS / hipBLAS
    # / SYCL on accelerators, Eigen on the host) on the packed site-major
    # layout; the input is packed into a scratch buffer since the batched
    # LU overwrites it.  As the former Eigen code, compute in double precision
    # also for single-precision fields.
    if A.grid.precision is not gpt.double:
        A_double = gpt.convert(A, gpt.double)
        target_double = _batched(A_double, gpt.lattice(A_double.grid, target.otype), job)
        gpt.convert(target, target_double)
        target.checkerboard(A.checkerboard())
        return target
    shape = A.otype.shape
    assert len(shape) == 2 or (len(shape) == 4 and shape[0] == shape[1] and shape[2] == shape[3])
    packed = gpt.pack(A).to_accelerator_buffer()
    sites = int(np.prod(packed.shape[: -len(shape)]))
    a = _as_matrices(packed, shape, sites)
    idx = np.arange(sites, dtype=np.int64)
    c = job(a, idx)
    if target.otype.shape == (1,):
        c = c.reshape((sites,))
    else:
        c = _from_matrices(c, shape, sites)
    gpt.pack(target).from_accelerator_buffer(c)
    target.checkerboard(A.checkerboard())
    return target


# pack does not support checkerboarded grids yet; these keep the per-site
# Eigen code on the host


def inv(A):
    A_inv = gpt.lattice(A)
    if A.grid.cb.n != 1:
        cgpt.invert_matrix([A_inv], [A])
        return A_inv

    def job(a, idx):
        c = a.empty_clone()
        gpt.accelerator.kernel().inv(a[idx], c[idx])()
        return c

    return _batched(A, A_inv, job)


def det(A):
    r = gpt.complex(A.grid)
    if A.grid.cb.n != 1:
        cgpt.determinant(r.v_obj[0], [A])
        return r.checkerboard(A.checkerboard())

    def job(a, idx):
        c = a.empty_clone(a.shape[0:1])
        gpt.accelerator.kernel().det(a[idx], c[idx])()
        return c

    return _batched(A, r, job)
