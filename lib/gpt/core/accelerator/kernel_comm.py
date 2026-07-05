#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025-26  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


def copy(self, plan, dst, src):
    dst = g.util.to_list(dst)
    src = g.util.to_list(src)
    self.references.append((plan, dst, src))
    cgpt.kernel_copy(self.obj, plan.obj, dst, src, g.accelerator)
    return self


def expand_to_global_and_transpose(self, dst, src, buffer_dimension, grid, grid_dimension):
    nd = len(dst.shape)
    assert nd == len(src.shape)
    assert dst.shape == tuple(
        [src.shape[i] for i in range(nd) if i != buffer_dimension]
        + [src.shape[buffer_dimension] * grid.mpi[grid_dimension]]
    )
    assert dst.dtype == src.dtype

    block_size = dst.calculate_size((1,), dst.dtype)

    dst_coor = dst.coordinates(range(nd))
    dst_stride = dst.strides()
    dst_idx = dst_coor @ dst_stride

    idx_all = list(range(nd))

    src_coor_global = dst_coor.take(
        idx_all[0:buffer_dimension] + [nd - 1] + idx_all[buffer_dimension:-1], axis=1
    )
    src_coor = np.mod(src_coor_global, src.shape)
    src_mpi_coor = src_coor_global // src.shape

    mpi_shift = src_mpi_coor[:, buffer_dimension] - grid.processor_coor[grid_dimension]
    src_rank = grid.processor + mpi_shift * grid.mpi_strides()[grid_dimension]
    src_stride = src.strides()

    src_idx = src_coor @ src_stride

    dst_view_idx = np.zeros(shape=(len(dst_idx), 4), dtype=np.int64)
    src_view_idx = np.zeros(shape=(len(src_idx), 4), dtype=np.int64)

    dst_view_idx[:, 0] = grid.processor
    src_view_idx[:, 0] = src_rank

    dst_view_idx[:, 2] = block_size * dst_idx
    src_view_idx[:, 2] = block_size * src_idx

    dst_view_idx[:, 3] = block_size
    src_view_idx[:, 3] = block_size

    cp = g.copy_plan(dst, src)
    cp.destination += g.global_memory_view(grid, dst_view_idx)
    cp.source += g.global_memory_view(grid, src_view_idx)
    cp = cp()
    copy(self, cp, dst, src)
    return self


def restrict_to_local_and_transpose(self, dst, src, buffer_dimension, grid, grid_dimension):
    nd = len(dst.shape)
    assert nd == len(src.shape)
    assert src.shape == tuple(
        [dst.shape[i] for i in range(nd) if i != buffer_dimension]
        + [dst.shape[buffer_dimension] * grid.mpi[grid_dimension]]
    )
    assert dst.dtype == src.dtype

    block_size = dst.calculate_size((1,), dst.dtype)

    dst_coor = dst.coordinates(range(nd))
    dst_stride = dst.strides()
    dst_idx = dst_coor @ dst_stride

    idx_all = list(range(nd))
    idx_all = idx_all[0:buffer_dimension] + [nd - 1] + idx_all[buffer_dimension:-1]
    idx_all = [idx_all.index(i) for i in range(nd)]

    src_coor = dst_coor + np.array(
        [
            (
                0
                if i != buffer_dimension
                else dst.shape[buffer_dimension] * grid.processor_coor[grid_dimension]
            )
            for i in range(nd)
        ]
    )
    src_coor = src_coor.take(idx_all, axis=1)

    src_stride = src.strides()
    src_idx = src_coor @ src_stride

    dst_view_idx = np.zeros(shape=(len(dst_idx), 4), dtype=np.int64)
    src_view_idx = np.zeros(shape=(len(src_idx), 4), dtype=np.int64)

    dst_view_idx[:, 0] = grid.processor
    src_view_idx[:, 0] = grid.processor

    dst_view_idx[:, 2] = block_size * dst_idx
    src_view_idx[:, 2] = block_size * src_idx

    dst_view_idx[:, 3] = block_size
    src_view_idx[:, 3] = block_size

    cp = g.copy_plan(dst, src)
    cp.destination += g.global_memory_view(grid, dst_view_idx)
    cp.source += g.global_memory_view(grid, src_view_idx)
    cp = cp()
    copy(self, cp, dst, src)
    return self
