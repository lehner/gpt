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
    def __init__(self, vdst, vsrc, communication_buffer = "host"): # host/accelerator/none
        self.obj = cgpt.copy_create_plan(vdst, vsrc, communication_buffer)

    def __del__(self):
        cgpt.copy_delete_plan(self.obj)

    def __call__(self, dst, src):
        # prepare? lattice vobjs
        cgpt.copy_execute_plan(self.obj, dst, src)

        
class copy_view:
    def __init__(self, obj):
        self.obj = obj

    def __del__(self):
        cgpt.copy_delete_view(self.obj)

    def __len__(self):
        return cgpt.copy_view_size(self.obj)
