#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import numpy as np
from gpt.core.contract.linear_map import linear_map


class indexed_sum(linear_map):
    def __init__(self, index, length):
        self.shape = (length,) + index.shape
        self.index = index
        self.cache = {}
        assert len(index.shape) == 1

    def __str__(self):
        return f"indexed_sum({self.shape}, {self.index.shape})"

    def commit_single_contract_after_trace(
        self, traced_source_buffer, target_buffer, dimension, kernel, bm
    ):

        n = len(traced_source_buffer.shape)
        S1 = [traced_source_buffer.shape[i] for i in range(1, n) if i != dimension]
        S2 = [target_buffer.shape[i] for i in range(1, n) if i != dimension]
        assert S1 == S2

        # now create index buffer
        traced_source_shape = traced_source_buffer.shape
        index_tag = tuple(traced_source_shape)
        if index_tag not in self.cache:
            strides = traced_source_buffer.strides()
            coor = traced_source_buffer.coordinates(range(n))
            coor[:, dimension] = self.index[coor[:, dimension]]
            indices = coor @ strides

            assert 0 <= np.min(indices)
            assert np.max(indices) < np.prod(target_buffer.shape)

            buffer_index = g.accelerator.buffer(
                shape=(len(indices),), dtype=np.dtype(self.index.dtype).type
            )
            buffer_index.from_array(indices)
            self.cache[index_tag] = buffer_index
        else:
            buffer_index = self.cache[index_tag]

        traced_source_buffer.flatten()

        target_buffer_shape = target_buffer.shape
        target_buffer.reshape((int(np.prod(target_buffer_shape)),))
        kernel.indexed_sum(traced_source_buffer, buffer_index, target_buffer)
        target_buffer.reshape(target_buffer_shape)
