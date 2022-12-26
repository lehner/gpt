#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt, numpy

# Needs: local_grid, lcoor, gcoor
class two_grid_base:
    def __init__(self):
        self.project_plan = {}
        self.promote_plan = {}

    def lattice(self, otype):
        return gpt.lattice(self.local_grid, otype)

    def project(self, dst, src):
        tag = src.otype.__name__
        if tag not in self.project_plan:
            plan = gpt.copy_plan(dst, src, embed_in_communicator=src.grid)
            plan.destination += dst.view[self.lcoor]
            plan.source += src.view[self.gcoor]
            self.project_plan[tag] = plan()
        self.project_plan[tag](dst, src)

    def promote(self, dst, src):
        tag = src.otype.__name__
        if tag not in self.promote_plan:
            plan = gpt.copy_plan(dst, src, embed_in_communicator=dst.grid)
            plan.destination += dst.view[self.gcoor]
            plan.source += src.view[self.lcoor]
            self.promote_plan[tag] = plan()
        self.promote_plan[tag](dst, src)
