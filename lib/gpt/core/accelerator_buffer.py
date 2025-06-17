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
import gpt, cgpt
import numpy as np


class accelerator_buffer_view:
    op_N = 0
    op_T = 1
    op_C = 2

    def __init__(self, buffer, idx, op=op_N):
        self.buffer = buffer
        self.idx = idx
        self.op = op

    @property
    def T(self):
        return accelerator_buffer_view(self.buffer, self.idx, self.op ^ self.op_T)

    @property
    def H(self):
        return accelerator_buffer_view(self.buffer, self.idx, self.op ^ (self.op_T | self.op_C))


class accelerator_buffer:
    def __init__(self, nbytes=None, shape=None, dtype=None):

        if isinstance(nbytes, accelerator_buffer):
            self.view = nbytes.view
            self.shape = nbytes.shape
            self.dtype = nbytes.dtype
            return

        if nbytes is None:
            nbytes = self.calculate_size(shape, dtype)

        assert nbytes % 4 == 0
        self.view = cgpt.create_device_memory_view(nbytes)
        if dtype is None:
            dtype = np.int8
            shape = (len(self.view),)
        self.shape = shape
        self.dtype = dtype

    def empty_clone(self, shape=None):
        if shape is None:
            shape = self.shape
        nbytes = self.calculate_size(shape, self.dtype)
        return accelerator_buffer(nbytes, shape, self.dtype)

    def calculate_size(self, shape, dtype):
        sites = int(np.prod(shape))
        if dtype is np.complex64:
            sites *= 8
        elif dtype is np.complex128:
            sites *= 16
        elif dtype is np.int8:
            sites *= 1
        else:
            assert False
        return sites

    def merged_axes(self, axes0, axes1):
        if axes0 < 0:
            axes0 += len(self.shape)
        if axes1 < 0:
            axes1 += len(self.shape)
        shape = list(self.shape)
        shape = tuple(
            shape[0:axes0] + [int(np.prod(shape[axes0 : axes1 + 1]))] + shape[axes1 + 1 :]
        )
        copy = accelerator_buffer(self)
        copy.shape = shape
        return copy

    def __getitem__(self, idx):
        return accelerator_buffer_view(self, idx)

    def check_size(self):
        if self.shape is not None and self.dtype is not None:
            assert len(self.view) == self.calculate_size(self.shape, self.dtype)

    def __str__(self):
        return f"accelerator_buffer({len(self.view)}, {self.shape}, {self.dtype.__name__})"

    def to_array(self):
        self.check_size()

        array = cgpt.ndarray(self.shape, self.dtype)
        cgpt.transfer_array_device_memory_view(array, self.view, True)

        return array

    def from_array(self, array):
        cgpt.transfer_array_device_memory_view(array, self.view, False)
        return self

    def coordinates(self, dimensions):
        dimensions = list(dimensions)
        nd = len(dimensions)
        L = [self.shape[dimensions[i]] for i in range(nd)]
        args = [slice(0, L[i]) for i in range(nd)]
        return np.mgrid[tuple(args)].reshape(nd, -1).T

    def indices(self, dimensions, shift=None):
        dimensions = list(dimensions)
        nd = len(dimensions)
        L = np.array([self.shape[dimensions[i]] for i in range(nd)], dtype=np.int64)
        idx0 = np.arange(np.prod(L))
        if shift is None:
            return idx0
        else:
            shift = np.mod(np.array(shift, dtype=np.int64) + L, L)
            stride = np.array([np.prod(L[i + 1 :]) for i in range(nd)])
            idx0 = idx0.reshape(idx0.shape[0], 1)
            idx1 = np.sum(np.mod(idx0 // stride + shift, L) * stride, axis=1)
            return idx1
            # idx = ((coor[i] + shift[i] + L[i]) % L[i]) * stride[i]
            #     = (((idx0 // stride[i] % L[i]) + shift[i] + L[i]) % L[i]) * stride[i]
            #     = ((idx0 // stride[i] + shift[i] + L[i]) % L[i]) * stride[i]

    def bulk(self, coordinates, margin=None, top_margin=None, bottom_margin=None):
        nd = len(coordinates[0])
        if margin is not None:
            top_margin = margin
            bottom_margin = margin
        if top_margin is None:
            top_margin = [0] * nd
        if bottom_margin is None:
            bottom_margin = [0] * nd

        top = np.array(top_margin, dtype=np.int64)
        bottom = np.array(
            [self.shape[i] - bottom_margin[i] - top_margin[i] for i in range(nd)], dtype=np.int64
        )

        return np.all(np.logical_and(coordinates >= top, coordinates <= bottom), axis=1)

    def halo_exchange(self, grid, margin, max_point_sqr=None):
        nd = len(margin)
        L = np.array([self.shape[i] - 2 * margin[i] for i in range(nd)], dtype=np.int64)
        idx = self.indices(range(nd))
        lc = self.coordinates(range(nd))

        mpi = list(reversed(grid.mpi))
        processor_stride = np.array([1] + list(np.cumprod(mpi[:-1])), dtype=np.int64)
        processor_coor = np.array(list(reversed(grid.processor_coor)), dtype=np.int64)

        G = L * mpi

        plan = gpt.copy_plan(self, self)

        my_rank = grid.processor

        block_size = self.calculate_size(self.shape[nd:], dtype=self.dtype)

        lcstride = np.array(
            list(reversed([1] + list(np.cumprod(list(reversed(self.shape[0:nd]))[0:-1])))),
            dtype=np.int64,
        )

        # assert np.linalg.norm(np.sum(lcstride * lc, axis=1) - idx) == 0

        def _mk_local(idx):
            dst = np.zeros(shape=(len(idx), 4), dtype=np.int64)
            dst[:, 0] = my_rank
            # dst[:,1] = 0
            dst[:, 2] = idx * block_size
            dst[:, 3] = block_size
            return gpt.global_memory_view(grid, dst)

        def _mk_remote(idx):
            gc = np.mod(lc[idx] - margin + processor_coor * L + G, G)
            mpc = gc // L
            ranks = mpc @ processor_stride
            mlc = gc - mpc * L + margin
            midx = np.sum(lcstride * mlc, axis=1)

            dst = np.zeros(shape=(len(idx), 4), dtype=np.int64)
            dst[:, 0] = ranks
            # dst[:,1] = 0
            dst[:, 2] = midx * block_size
            dst[:, 3] = block_size
            return gpt.global_memory_view(grid, dst)

        assert my_rank == processor_stride @ processor_coor

        t = gpt.timer("create halo exchange")
        for dim in range(nd):
            # get lower/upper planes
            t("mask")
            ortho_mask = np.full(shape=idx.shape, fill_value=True)
            if max_point_sqr == 1:
                for odim in range(nd):
                    if odim == dim:
                        continue
                    ortho_mask = np.logical_and(ortho_mask, lc[:, odim] >= margin[odim])
                    ortho_mask = np.logical_and(ortho_mask, lc[:, odim] < margin[odim] + L[odim])
            lower_margin = idx[np.logical_and(ortho_mask, lc[:, dim] < margin[dim])]
            upper_margin = idx[np.logical_and(ortho_mask, lc[:, dim] >= (L[dim] + margin[dim]))]

            t("plan view")
            plan.destination += _mk_local(lower_margin)
            plan.source += _mk_remote(lower_margin)

            plan.destination += _mk_local(upper_margin)
            plan.source += _mk_remote(upper_margin)

        t("plan create")
        plan = plan()
        t()

        def _caller():
            plan(self, self)

        return _caller
