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


class linear_map:
    def commit_single_contract(self, util, kernel, code, bm):
        assert len(code) == 3
        target, index, source = code
        assert len(index) == 3
        index_op, index_dst, index_src = index
        assert index_op is self

        # make sure we have target and source buffers
        assert isinstance(target[0], g.accelerator.buffer)
        assert isinstance(source[0], g.accelerator.buffer)

        target_buffer = target[0]
        target_indices = target[1:]

        source_buffer = source[0]
        source_indices = tuple(source[1:])

        assert target_buffer.dtype is source_buffer.dtype

        # first contract the source indices that are not index_src and not in target_indices
        assert index_src in source_indices
        assert index_dst not in source_indices
        assert index_dst in target_indices
        assert index_src not in target_indices

        traced_source_indices = tuple([x if x != index_dst else index_src for x in target_indices])

        if traced_source_indices == source_indices:
            self.commit_single_contract_after_trace(
                source_buffer, target_buffer, target_indices.index(index_dst), kernel, bm
            )
        else:

            traced_source_shape = util.shape_from_dimensions(traced_source_indices)
            traced_source_buffer = bm.request(shape=traced_source_shape, dtype=target_buffer.dtype)
            kernel.contract((traced_source_buffer, *traced_source_indices), source)

            self.commit_single_contract_after_trace(
                traced_source_buffer, target_buffer, target_indices.index(index_dst), kernel, bm
            )

            bm.release(traced_source_buffer)
        return True
