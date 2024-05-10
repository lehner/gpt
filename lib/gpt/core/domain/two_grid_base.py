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
import gpt as g


# Needs: local_grid, lcoor, gcoor
class two_grid_base:
    def __init__(self):
        self.project_plan = {}
        self.promote_plan = {}

    def lattice(self, otype):
        return g.lattice(self.local_grid, otype)

    def project(self, dst, src):
        dst = g.util.to_list(dst)
        src = g.util.to_list(src)
        tag = str([s.otype.__name__ for s in src])
        if tag not in self.project_plan:
            plan = g.copy_plan(dst, src, embed_in_communicator=src[0].grid)
            assert len(dst) == len(src)
            for i in range(len(dst)):
                plan.destination += dst[i].view[self.lcoor_project]
                plan.source += src[i].view[self.gcoor_project]
            self.project_plan[tag] = plan()
        self.project_plan[tag](dst, src)

    def promote(self, dst, src):
        dst = g.util.to_list(dst)
        src = g.util.to_list(src)
        tag = str([s.otype.__name__ for s in src])
        if tag not in self.promote_plan:
            plan = g.copy_plan(dst, src, embed_in_communicator=dst[0].grid)
            assert len(dst) == len(src)
            for i in range(len(dst)):
                plan.destination += dst[i].view[self.gcoor_promote]
                plan.source += src[i].view[self.lcoor_promote]
            self.promote_plan[tag] = plan()
        self.promote_plan[tag](dst, src)
