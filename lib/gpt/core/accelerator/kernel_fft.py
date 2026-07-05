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


def rank_fft(self, target, source, sign):
    # target_{*x} = e^{sign i (2pi/L) n x} source_{*n}
    # no normalization, acts on fastest running dimension
    assert source.shape == target.shape
    assert source.dtype is target.dtype
    howmany = int(np.prod(source.shape[0:-1]))
    size = source.shape[-1]
    cgpt.kernel_fft(self.obj, source.view, target.view, target.dtype, howmany, size, sign)
    return self


def fft(self, bm, target, source, dimension, grid, grid_dimension, forward=True):
    # will match gpt convention of having forward transformation with +1 sign
    assert source.shape == target.shape
    assert source.dtype is target.dtype
    nd = len(source.shape)
    temp_shape = tuple(
        [source.shape[i] for i in range(nd) if i != dimension]
        + [source.shape[dimension] * grid.mpi[grid_dimension]]
    )
    temp = bm.request(shape=temp_shape, dtype=target.dtype)
    temp2 = bm.request(shape=temp_shape, dtype=target.dtype)
    self.expand_to_global_and_transpose(temp, source, dimension, grid, grid_dimension)
    self.rank_fft(temp2, temp, +1 if forward else -1)
    self.restrict_to_local_and_transpose(target, temp2, dimension, grid, grid_dimension)
    if forward:
        self.accumulate(
            [target, target], np.array([1 / temp.shape[-1]], dtype=target.dtype), zero=True
        )
    bm.release(temp)
    bm.release(temp2)
    return self
