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
import gpt as g

default_padding_cache = {}


class padded_local_fields:
    def __init__(self, fields, margin_top, margin_bottom=None, cache=default_padding_cache):
        fields = g.util.to_list(fields)
        self.grid = fields[0].grid
        self.otype = fields[0].otype
        self.checkerboard = fields[0].checkerboard()
        self.padded_checkerboard = self.checkerboard

        if margin_bottom is None:
            margin_bottom = margin_top

        if self.checkerboard is not g.none:
            boundary_parity = sum(margin_top) % 2
            assert boundary_parity == sum(margin_bottom) % 2

        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        assert all([f.otype.__name__ == self.otype.__name__ for f in fields])
        assert all([f.grid.obj == self.grid.obj for f in fields])

        tag = f"{self.grid}_{self.checkerboard}_{margin_top}_{margin_bottom}"
        if tag not in cache:
            cache[tag] = g.domain.local(
                self.grid, margin_top, margin_bottom, cb=self.padded_checkerboard
            )
        self.domain = cache[tag]

    def __call__(self, fields):
        return_list = isinstance(fields, list)
        fields = g.util.to_list(fields)
        padded_fields = [self.domain.lattice(self.otype) for f in fields]
        for p in padded_fields:
            p.checkerboard(self.checkerboard)
        self.domain.project(padded_fields, fields)
        return padded_fields if return_list else padded_fields[0]

    def extract(self, fields, padded_fields):
        self.domain.promote(fields, padded_fields)
