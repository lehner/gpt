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


class ot_base:
    v_otype = [None]  # cgpt's data types
    v_n0 = [0]
    v_n1 = [1]
    v_idx = [0]
    transposed = None
    spintrace = None
    colortrace = None
    data_alias = None  # ot can be cast as fundamental type data_alias (such as SU(3) -> 3x3 matrix)
    mtab = {}  # x's multiplication table for x * y
    rmtab = {}  # y's multiplication table for x * y

    # only vectors shall define otab/itab
    otab = None  # x's outer product multiplication table for x * adj(y)
    itab = None  # x's inner product multiplication table for adj(x) * y

    # list of object types to which I can convert and converter function
    ctab = {}

    # safe cast of data_alias
    def data_otype(self):
        if self.data_alias is not None:
            return self.data_alias()
        return self
