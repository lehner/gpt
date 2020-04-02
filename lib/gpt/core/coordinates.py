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
import gpt
import cgpt

def coordinates(o):
    if type(o) == gpt.grid:
        dim=len(o.ldimensions)
        top=[ o.processor_coor[i]*o.ldimensions[i] for i in range(dim) ]
        bottom=[ top[i] + o.ldimensions[i] for i in range(dim) ]
        return cgpt.coordinates_form_cartesian_view(top,bottom)
    elif type(o) == gpt.cartesian_view:
        return cgpt.coordinates_form_cartesian_view(o.top,o.bottom)
    else:
        assert(0)
