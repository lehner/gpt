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

def propagator(inv_matrix, w = None):

    if w is None:
        exp = inv_matrix.ExportPhysicalFermionSolution
        imp = inv_matrix.ImportPhysicalFermionSource
    else:
        exp = w.ExportPhysicalFermionSolution
        imp = w.ImportPhysicalFermionSource

    def prop(dst_sc, src_sc):
        dst_sc @= exp * inv_matrix * imp * src_sc

    r=gpt.matrix_operator(prop, 
                          otype = (exp.otype[0],imp.otype[1]),
                          grid = (exp.grid[0],imp.grid[1]))

    r.inv_matrix = inv_matrix

    return r
