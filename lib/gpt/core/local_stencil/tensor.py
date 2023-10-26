#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import cgpt


def parse(c):
    if isinstance(c, tuple):
        assert len(c) == 5
        return {
            "target": c[0],
            "element": c[1],
            "instruction": c[2],
            "weight": c[3],
            "factor": c[4],
        }
    return c


class tensor:
    def __init__(self, lat, points, code, segments, local=1):
        self.points = points
        self.code = [parse(c) for c in code]
        self.segments = segments
        self.obj = cgpt.stencil_tensor_create(
            lat.v_obj[0], lat.grid.obj, points, self.code, self.segments, local
        )
        self.osites_per_instruction = 4
        self.osites_per_cache_block = lat.grid.gsites

    def __call__(self, *fields):
        cgpt.stencil_tensor_execute(self.obj, list(fields),
                                    self.osites_per_instruction,
                                    self.osites_per_cache_block)

    def __del__(self):
        cgpt.stencil_tensor_delete(self.obj)

    def data_access_hints(self, *hints):
        pass

    def memory_access_pattern(self, osites_per_instruction, osites_per_cache_block):
        self.osites_per_instruction = osites_per_instruction
        self.osites_per_cache_block = osites_per_cache_block
