#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann (daniel.richtmann@ur.de)
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
import cgpt, gpt, numpy


class lookup_table:
    def __init__(self, coarse_grid, second):
        assert type(coarse_grid) == gpt.grid
        if type(second) == gpt.grid:
            mask = gpt.vcomplex(second)
            mask[:] = 1.0
            gpt.make_lut(coarse_grid, mask)
        elif type(second) == gpt.lattice:
            # TODO: need some form of caching here!
            assert len(second.v_obj) == 1
            self.obj = cgpt.create_lookup_table(coarse_grid.obj, second.v_obj[0])
        else:
            raise Exception("Unknown lookup_table constructor")

    def __del__(self):
        cgpt.delete_lookup_table(self.obj)
