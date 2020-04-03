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
import gpt
import gpt.core.io.gpt_io

# expose fast memoryview for numpy arrays
def mview(data):
    mv=cgpt.mview(data)
    assert(mv.obj is data)
    return mv

# file formats
class format:
    class gpt:
        def __init__(self, params = {}):
            self.params = params
    class cevec:
        def __init__(self, params = {}):
            self.params = params

# load through cgpt backend (NerscIO, openQCD, cevec, ...)
def load_cgpt(*a):
    result=[]
    r,metadata=cgpt.load(*a, gpt.default.is_verbose("io"))
    for gr in r:
        grid=gpt.grid(gr[1],eval("gpt.precision." + gr[2]),eval("gpt." + gr[3]),gr[0])
        result_grid=[]
        for t_obj,s_ot,s_pr in gr[4]:
            assert(s_pr == gr[2])
            l=gpt.lattice(grid,eval("gpt.otype." + s_ot),t_obj)
            l.metadata=metadata
            result_grid.append(l)
        result.append(result_grid)
    while len(result) == 1:
        result=result[0]
    return result

# input
def load(*a):

    try:
        return gpt_io.load(*a)
    except NotImplementedError:
        pass

    return load_cgpt(*a)

# output
def save(filename,objs,fmt = format.gpt()):

    if type(fmt) == format.gpt:
        return gpt_io.save(filename, objs, fmt.params)

    return cgpt.save(filename, objs, fmt, gpt.default.is_verbose("io"))

# helper
def crc32(view):
    return cgpt.util_crc32(view)
