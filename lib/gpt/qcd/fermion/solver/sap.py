#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Mattia Bruno
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

# This inverter approximates the solution of the Dirac equation
# using the Schwartz Alternating Procedure as described here
#
# M. Luescher, "Solution of the Dirac equation in lattice QCD using a 
#               domain decomposition method"
# https://arxiv.org/abs/hep-lat/0310048
#
# It is based on the sap preconditioner class that contains the
# basic functionalities relevant here.
#
# The code below is a first working version; improvements for
# better performances could be achieved by substituting the
# application of op.M with the simpler hopping from even/odd
# blocks.
import gpt

def inv_sap(sap, blk_solver, ncy):
    
    otype = sap.op.otype
    src_blk = gpt.lattice(sap.op_blk[0].F_grid, otype)
    dst_blk = gpt.lattice(sap.op_blk[0].F_grid, otype)
    solver = [blk_solver(op) for op in sap.op_blk]
    
    def inv(psi, rho):
        psi[:] = 0
        eta = gpt.copy(rho)
        ws = [gpt.copy(rho) for _ in range(2)]
        
        for ic in range(1,ncy+1):
            dt_solv=dt_distr=dt_hop=0.0
            for eo in range(2):
                ws[0][:] = 0
                dt_distr-=gpt.time()
                src_blk[sap.pos] = eta[sap.coor[eo]]
                dt_distr+=gpt.time()
                
                dt_solv-=gpt.time()
                solver[eo](dst_blk, src_blk)
                dt_solv+=gpt.time()
                
                dt_distr-=gpt.time()
                ws[0][sap.coor[eo]] = dst_blk[sap.pos]
                dt_distr+=gpt.time()
                
                dt_hop-=gpt.time()
                sap.op.M(ws[1], ws[0])
                eta -= ws[1]
                psi += ws[0]
                dt_hop+=gpt.time()
                
            gpt.message(f'SAP cycle = {ic}; |rho|^2 = {gpt.norm2(eta):g}; |psi|^2 = {gpt.norm2(psi):g}')
            gpt.message(f'SAP Timings: distr {dt_distr:g} secs, blk_solver {dt_solv:g} secs, hop+update {dt_hop:g} secs')
                
    m = gpt.matrix_operator(
        mat=inv,
        inv_mat=sap.op.M,
        adj_inv_mat=sap.op.M.adj(),
        adj_mat=None,  # implement adj_mat when needed
        otype=otype,
        zero=(True, False),
        grid=sap.op.F_grid,
        cb=None,
    )

    m.ImportPhysicalFermionSource = sap.op.ImportPhysicalFermionSource
    m.ExportPhysicalFermionSolution = sap.op.ExportPhysicalFermionSolution

    return m        