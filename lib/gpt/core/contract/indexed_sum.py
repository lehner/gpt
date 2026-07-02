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

    def commit_single_contract_after_trace(self, traced_source_buffer, target_buffer, kernel):
        # now create index buffer
        traced_source_shape = traced_source_buffer.shape
        stride = int(np.prod(traced_source_shape[1:]))
        tsize = int(np.prod(traced_source_shape))

        index_tag = (stride, tsize)
        if index_tag not in self.cache:
            buffer_index = g.accelerator.buffer(
                shape=(tsize,), dtype=np.dtype(self.index.dtype).type
            )
            buffer_index_array = np.repeat(self.index, stride) * stride + np.tile(
                np.arange(stride), tsize // stride
            )
            assert 0 <= np.min(buffer_index_array)
            assert np.max(buffer_index_array) < np.prod(target_buffer.shape)
            buffer_index.from_array(buffer_index_array)
            self.cache[index_tag] = buffer_index
        else:
            buffer_index = self.cache[index_tag]

        traced_source_buffer.reshape((tsize,))

        target_buffer_shape = target_buffer.shape
        target_buffer.reshape((int(np.prod(target_buffer_shape)),))
        kernel.indexed_sum(traced_source_buffer, buffer_index, target_buffer)
        target_buffer.reshape(target_buffer_shape)
