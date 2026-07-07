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


class fft(linear_map):
    def __init__(self, dimension, grid, forward=True, grid_dimension=None):
        if grid_dimension is None:
            grid_dimension = grid.nd - dimension - 1
        self.grid_dimension = grid_dimension
        self.grid = grid
        self.forward = forward
        self.shape = (grid.ldimensions[grid_dimension], grid.ldimensions[grid_dimension])

    def __str__(self):
        return f"fft({self.shape})"

    def commit_single_contract_after_trace(
        self, traced_source_buffer, target_buffer, dimension, kernel, bm
    ):
        assert traced_source_buffer.shape == target_buffer.shape

        kernel.fft(
            bm,
            target_buffer,
            traced_source_buffer,
            dimension,
            self.grid,
            self.grid_dimension,
            self.forward,
        )
