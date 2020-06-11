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

def inv_eo_ne(matrix, inverter):

    F_grid_eo=matrix.F_grid_eo
    otype=matrix.otype

    ie=gpt.lattice(F_grid_eo,otype)
    io=gpt.lattice(F_grid_eo,otype)
    t1=gpt.lattice(F_grid_eo,otype)
    t2=gpt.lattice(F_grid_eo,otype)
    
    def inv(dst_sc, src_sc):

        oe=gpt.lattice(F_grid_eo,otype)
        oo=gpt.lattice(F_grid_eo,otype)

        gpt.pick_cb(gpt.even,ie,src_sc)
        gpt.pick_cb(gpt.odd,io,src_sc)

        # D^-1 = L NDagN^-1 R + S

        matrix.R(t1, ie, io)

        t2[:]=0
        t2.checkerboard(gpt.even)

        inverter(matrix.NDagN)(t2,t1)

        matrix.L(oe, oo, t2)

        matrix.S(t1,t2,ie,io)

        oe += t1
        oo += t2

        gpt.set_cb(dst_sc,oe)
        gpt.set_cb(dst_sc,oo)

    m=gpt.matrix_operator(mat = inv, inv_mat = matrix.op.M,
                          adj_inv_mat = matrix.op.M.adj(),
                          adj_mat = None, # implement adj_mat when needed
                          otype = otype, zero_lhs = True)
    
    m.ImportPhysicalFermionSource = matrix.ImportPhysicalFermionSource
    m.ExportPhysicalFermionSolution = matrix.ExportPhysicalFermionSolution

    return m
