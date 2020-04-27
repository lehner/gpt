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
import cgpt

# expose fast memoryview for numpy arrays
def mview(data):
    mv=cgpt.mview(data)
    assert(mv.obj is data)
    return mv

# fast threaded checksum of memoryviews
def crc32(view, crc32_prev = 0):
    if type(view) == memoryview:
        return cgpt.util_crc32(view, crc32_prev)
    else:
        return crc32(memoryview(view), crc32_prev)

