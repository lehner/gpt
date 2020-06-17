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
        t2.checkerboard(t1.checkerboard())

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
                          otype = otype, zero = (True,False), grid = matrix.F_grid,
                          cb = None)
    
    m.ImportPhysicalFermionSource = matrix.ImportPhysicalFermionSource
    m.ExportPhysicalFermionSolution = matrix.ExportPhysicalFermionSolution

    return m



class a2a_eo_ne:

    def __init__(self, matrix):
        self.matrix = matrix
        self.F_grid_eo=matrix.F_grid_eo
        self.F_grid=matrix.F_grid
        self.U_grid=matrix.U_grid
        self.otype=matrix.otype

        self.oe=gpt.lattice(self.F_grid_eo,self.otype)
        self.oo=gpt.lattice(self.F_grid_eo,self.otype)
        self.U_tmp=gpt.lattice(self.U_grid,self.otype)
        self.F_tmp=gpt.lattice(self.F_grid,self.otype)
        self.F_tmp_2=gpt.lattice(self.F_grid,self.otype)

        def _v_unphysical(dst, evec):
            self.matrix.L(self.oe, self.oo, evec)
            gpt.set_cb(dst,self.oe)
            gpt.set_cb(dst,self.oo)

        def _w_unphysical(dst, evec):
            self.matrix.RDag(self.oe, self.oo, evec)
            gpt.set_cb(dst,self.oe)
            gpt.set_cb(dst,self.oo)
            
        def _v(dst, evec):
            _v_unphysical(self.F_tmp,evec)
            self.matrix.ExportPhysicalFermionSolution(dst,self.F_tmp)

        def _w(dst, evec):
            _w_unphysical(self.F_tmp,evec)
            self.matrix.Dminus.adj_mat(self.F_tmp_2,self.F_tmp)
            self.matrix.ExportPhysicalFermionSource(dst,self.F_tmp_2)

        def _G5w(dst, evec):
            _w(self.U_tmp, evec)
            dst @= gpt.gamma[5] * self.U_tmp

        self.v=gpt.matrix_operator(mat = _v,
                                   otype = self.otype, zero = (False,False), 
                                   grid = (self.U_grid,self.F_grid_eo))

        self.w=gpt.matrix_operator(mat = _w,
                                   otype = self.otype, zero = (False,False), 
                                   grid = (self.U_grid,self.F_grid_eo))

        self.G5w=gpt.matrix_operator(mat = _G5w,
                                   otype = self.otype, zero = (False,False), 
                                   grid = (self.U_grid,self.F_grid_eo))
        
        self.v_unphysical=gpt.matrix_operator(mat = _v_unphysical,
                                              otype = self.otype, zero = (False,False), 
                                              grid = (self.F_grid,self.F_grid_eo))

        self.w_unphysical=gpt.matrix_operator(mat = _w_unphysical,
                                              otype = self.otype, zero = (False,False), 
                                              grid = (self.F_grid,self.F_grid_eo))
