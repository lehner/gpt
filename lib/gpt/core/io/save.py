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
import gpt, cgpt
from gpt.params import params_convention


# format
class format:
    class gpt:
        @params_convention(mpi=None)
        def __init__(self, params):
            self.params = params

    class cevec:
        @params_convention(nsingle=None, max_read_blocks=None, mpi=None)
        def __init__(self, params):
            self.params = params

    class nersc:
        @params_convention(label="", id="gpt", sequence_number=1)
        def __init__(self, params):
            self.params = params


# output
def save(filename, objs, fmt=format.gpt()):
    if isinstance(fmt, format.gpt):
        return gpt.core.io.gpt_io.save(filename, objs, fmt.params)
    elif isinstance(fmt, format.cevec):
        return gpt.core.io.cevec_io.save(filename, objs, fmt.params)

    return cgpt.save(filename, objs, fmt, gpt.default.is_verbose("io"))
