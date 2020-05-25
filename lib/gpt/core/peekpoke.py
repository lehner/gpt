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
import gpt, cgpt, sys

def peek(target,key):

    if type(target) == gpt.lattice:
        return gpt.mview(target[key])

    elif type(target) == list:
        v_obj=[ y for x in target for y in x.v_obj ]
        return gpt.mview(cgpt.lattice_export(v_obj, key))

    else:
        assert(0)


def poke(target,key,value):

    assert(type(value) == memoryview)

    if type(target) == gpt.lattice:
        target[key]=value
    elif type(target) == list:
        v_obj=[ y for x in target for y in x.v_obj ]
        cgpt.lattice_import(v_obj, key, value)
    else:
        assert(0)

