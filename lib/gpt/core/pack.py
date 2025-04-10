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
from gpt.core import auto_tuned_class, auto_tuned_method
import numpy as np
import cgpt


def margin_to_padding_offset(margin, top_margin, bottom_margin):
    if margin is not None:
        top_margin = margin
        bottom_margin = margin
    if top_margin is not None and bottom_margin is not None:
        return [x + y for x, y in zip(top_margin, bottom_margin)], top_margin
    return None, None


class pack(auto_tuned_class):
    def __init__(self, lattices, fast=False):
        self.lattices = g.util.to_list(lattices)
        self.otype = self.lattices[0].otype
        self.grid = self.lattices[0].grid
        self.words = int(np.prod(self.otype.shape))
        if not fast:
            assert all([l.otype.__name__ == self.otype.__name__ for l in self.lattices])
            assert all([l.grid.obj == self.grid.obj for l in self.lattices])

        # auto tuner
        tag = f"pack({self.otype.__name__}, {len(self.lattices)}, {self.grid.describe()})"
        super().__init__(tag, [2, 4, 8, 16, 32, 64, 128, 256], 32)

    def rank_bytes(self):
        return sum([l.rank_bytes() for l in self.lattices])

    def allocate_accelerator_buffer(self, padding=None):
        shape = (
            list(reversed(self.grid.ldimensions)) + [len(self.lattices)] + list(self.otype.shape)
        )
        bytes_scale_num = 1
        bytes_scale_denom = 1
        if padding is not None:
            bytes_scale_denom = int(np.prod(shape))
            if len(padding) < len(shape):
                padding = padding + [0] * (len(shape) - len(padding))
            shape = [x + y for x, y in zip(shape, padding)]
            bytes_scale_num = int(np.prod(shape))
        return g.accelerator_buffer(
            (self.rank_bytes() * bytes_scale_num) // bytes_scale_denom,
            tuple(shape),
            self.grid.precision.complex_dtype,
        )

    def to_accelerator_buffer(
        self, target_buffer=None, margin=None, top_margin=None, bottom_margin=None
    ):
        padding, offset = margin_to_padding_offset(margin, top_margin, bottom_margin)
        if target_buffer is None:
            target_buffer = self.allocate_accelerator_buffer(padding=padding)
        self.transfer_accelerator_buffer(target_buffer, True, padding, offset)
        return target_buffer

    def from_accelerator_buffer(self, buffer, margin=None, top_margin=None, bottom_margin=None):
        padding, offset = margin_to_padding_offset(margin, top_margin, bottom_margin)
        self.transfer_accelerator_buffer(buffer, False, padding, offset)

    @auto_tuned_method(skip_snapshot=True)
    def transfer_accelerator_buffer(self, threads, buffer, export, padding, offset):
        buf = buffer.view
        r = self.otype.v_rank

        cgpt.lattice_transfer_scalar_device_buffer(
            self.lattices, buf, padding, offset, r, 1 if export else 0, threads
        )
