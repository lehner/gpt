#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
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


class cshift_executer:
    def __init__(self, buffer_descriptions, plan):
        self.buffer_descriptions = buffer_descriptions
        self.plan = plan

    def __call__(self, first, second=None):
        if second is None:
            fields = first
            buffers = [g.lattice(b[0], b[1]) for b in self.buffer_descriptions]
        else:
            buffers = g.util.to_list(first)
            fields = second

        for i in range(len(self.buffer_descriptions)):
            buffers[i].checkerboard(self.buffer_descriptions[i][2])
        self.plan(buffers, fields)
        return buffers


class cshift_plan:
    def __init__(self):
        self.displacements = []
        self.sources = []
        self.destinations = []
        self.index = 0
        self.indices = []

    def add(self, field, displacements):
        self.sources.append(field)
        self.displacements.append(displacements)
        indices = {}
        for d in displacements:
            indices[d] = self.index
            dst = g.lattice(field)
            self.destinations.append(dst)

            if dst.grid.cb.n == 2:
                if sum(d) % 2 != 0:
                    dst.checkerboard(field.checkerboard().inv())

            self.index += 1
        self.indices.append(indices)
        return indices

    def __call__(self):
        plan = g.copy_plan(self.destinations, self.sources)
        buffer_descriptions = []
        for i in range(len(self.sources)):
            src = self.sources[i]
            coordinates = g.coordinates(src)
            L = src.grid.fdimensions
            for x in self.displacements[i]:
                dst = self.destinations[self.indices[i][x]]
                buffer_descriptions.append((src.grid, src.otype, dst.checkerboard()))
                plan.destination += dst.view[
                    cgpt.coordinates_shift(coordinates, tuple([-y for y in x]), L)
                ]
                plan.source += src.view[:]
        return cshift_executer(buffer_descriptions, plan())
