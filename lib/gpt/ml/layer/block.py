#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.ml.layer import base_no_bias


class project(base_no_bias):
    def __init__(self, block_map):
        self.block_map = block_map
        super().__init__(None, None, None, 0)

    def __call__(self, weights, layer_input):
        return self.block_map.project(layer_input)

    def projected_gradient_adj(self, weights, layer_input, left):
        return [self.block_map.project.adj()(left)]


class promote(base_no_bias):
    def __init__(self, block_map):
        self.block_map = block_map
        super().__init__(None, None, None, 0)

    def __call__(self, weights, layer_input):
        return self.block_map.promote(layer_input)

    def projected_gradient_adj(self, weights, layer_input, left):
        return [self.block_map.promote.adj()(left)]
