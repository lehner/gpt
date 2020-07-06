#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann (daniel.richtmann@ur.de)
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


def inv_direct(matrix, inverter):
    def inv(dst_sc, src_sc):
        dst_sc @= inverter(matrix.M) * src_sc

    m = gpt.matrix_operator(
        mat=inv, inv_mat=matrix, otype=matrix.otype, grid=matrix.F_grid
    )

    m.ImportPhysicalFermionSource = matrix.ImportPhysicalFermionSource
    m.ExportPhysicalFermionSolution = matrix.ExportPhysicalFermionSolution

    return m
