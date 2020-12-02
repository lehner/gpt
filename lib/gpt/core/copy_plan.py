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
import cgpt, gpt, numpy

class copy_plan:
    def __init__(self, vdst, vsrc, lattice_view_location = "host", communication_buffer_location = "host"): # host/accelerator/none
        self.obj = cgpt.copy_create_plan(vdst.obj, vsrc.obj, communication_buffer_location)
        self.lattice_view_location = lattice_view_location

    def __del__(self):
        cgpt.copy_delete_plan(self.obj)

    def __call__(self, dst, src):
        dst = gpt.util.to_list(dst)
        src = gpt.util.to_list(src)
        cgpt.copy_execute_plan(self.obj, dst, src, self.lattice_view_location)

        
class copy_view:
    def __init__(self, first, second = None):
        if second is None:
            self.obj = first
        else:
            grid_obj = first
            blocks = numpy.array(second, dtype=numpy.int64)
            self.obj = cgpt.copy_create_view(grid_obj, blocks)

    def __del__(self):
        cgpt.copy_delete_view(self.obj)

    def __len__(self):
        return cgpt.copy_view_size(self.obj)
