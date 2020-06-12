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

def inv_g5m_ne(matrix, inverter):
    F_grid=matrix.F_grid
    ftmp=gpt.vspincolor(F_grid)
    i = inverter(lambda o,i: (matrix.G5M(ftmp,i),matrix.G5M(o,ftmp)))
    
    def inv(dst_sc, src_sc):
        #(G5 M G5 M)^-1 G5 M G5 = M^-1 G5 M^-1 G5^2 M G5 = M^-1
        dst_sc @= i * matrix.G5M * gpt.gamma[5] * src_sc

    m = gpt.matrix_operator(mat = inv, inv_mat = matrix, otype=matrix.otype,
                            grid = F_grid)

    m.ImportPhysicalFermionSource = matrix.ImportPhysicalFermionSource
    m.ExportPhysicalFermionSolution = matrix.ExportPhysicalFermionSolution
        
    return m
