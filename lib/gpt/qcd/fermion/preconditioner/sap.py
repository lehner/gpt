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

# This preconditioner performs a block decomposition of the local
# lattice; even and odd blocks are defined from their block id and
# and they are respectively packed together into two major blocks.
# The corresponding block operator is defined on the two clusters of
# blocks: links are properly set to zero to ensure Dirichelet Boundary
# Conditions, such that the operators are automatically defined on
# the blocks.
# The SAP class takes care of performing the block decomposition of
# the gauge field and of defining the corresponding block operators.
# It also contains lists of the coordinates corresponding to the even
# and odd blocks, such that fields can be easily assigned to blocks
# and viceversa.

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

#
#      ( EE EO )
#  M = ( OE OO )
#
#      ( EE^-1            0     )
#  K = ( -OO^-1 OE EE-1   OO^-1 )
#
#       ( 1 - EO OO^-1 OE EE-1    EO OO^-1  )
#  MK = ( 0                       1         )
#
#  Then K \sum_{n=0}^N (1 - MK)^n -> K (MK)^-1 = M^-1
#
#
#  eps = 1 - MK
#
#        ( EO OO^-1 OE EE-1       -EO OO^-1  )
#      = ( 0                       0         )
#
# This structure maps very well to our defect-correcting inverter:
#
#   outer_mat     =  (1 + defect) inner_mat
#
#   outer_mat^-1  = inner_mat^-1 \sum_{n=0}^N (- defect)^n
#
# with
#
#   inner_mat^-1 = K  and outer_mat = M
#
#   -defect = 1 - outer_mat inner_mat^{-1} = 1 - M K
#
import gpt, cgpt, numpy
from gpt.params import params_convention

# ix = x[0] + lat[0]*(x[1] + lat[2]*...)
def index2coor(ix, lat):  # go through cgpt and order=lexicographic
    x = [0] * len(lat)
    for mu in range(len(lat)):
        x[mu] = int(ix % lat[mu])
        ix = numpy.floor(ix / lat[mu])
    return x


class SapError(Exception):
    pass


class sap_blk:
    def __init__(self, grid, bs, eo):
        self.eo = eo
        self.bs = list(bs)
        Nd = len(bs)
        if Nd != grid.nd:
            raise SapError("Block dimensions do not match lattice dimensions")
        if numpy.any(self.bs > grid.ldimensions):
            raise SapError("Block size should not exceed local lattice")

        # block dimensions global
        self.bd = numpy.floor_divide(grid.fdimensions, self.bs)
        # block dimensions local
        self.bdl = numpy.floor_divide(grid.ldimensions, self.bs)

        # number of blocks per node
        self.nb = int(numpy.prod(self.bd)) // grid.Nprocessors
        if (self.nb < 2) or ((self.nb % 2) != 0):
            raise SapError("Sap Blocks should define an even/odd grid")

        # extended block sizes
        self.ebs = [1] * Nd
        for mu in reversed(range(Nd)):
            if self.bd[mu] > 1:
                self.ebs[mu] = int(self.nb / 2)
                break
        ebs = [bs[mu] * self.ebs[mu] for mu in range(Nd)]

        # block grid local to the node
        self.grid = grid.split([1] * Nd, ebs)

        self.bv = int(numpy.prod(self.bs))
        assert self.grid.gsites * 2 == (self.bv * self.nb)
        if self.bv < 4:
            raise SapError("Block volume should be bigger than 4")
        self.pos()
        gpt.message(
            f'SAP Initialized {"even" if self.eo==0 else "odd"} blocks with grid {self.grid.fdimensions} from local lattice {grid.ldimensions}'
        )

    def coor(self, grid, tag=None):  # and pos -> core/block/
        coor = numpy.zeros((self.grid.gsites, self.grid.nd), dtype=numpy.int32)

        n = 0
        # global offset of the local lattice
        ofs = [grid.processor_coor[mu] * grid.ldimensions[mu] for mu in range(grid.nd)]
        for ib in range(self.nb):
            bc = index2coor(ib, self.bdl)
            _eo = int(numpy.sum(bc) % 2)

            if _eo == self.eo:
                sl = slice(n * self.bv, (n + 1) * self.bv)
                n += 1

                top = [ofs[mu] + bc[mu] * self.bs[mu] for mu in range(len(ofs))]
                bottom = [top[mu] + self.bs[mu] for mu in range(len(ofs))]
                pos = cgpt.coordinates_from_cartesian_view(
                    top, bottom, grid.cb.cb_mask, tag, "lexicographic"
                )

                coor[sl, :] = pos
        assert n * 2 == self.nb
        return coor

    def pos(self, tag=None):
        self.pos = numpy.zeros((self.grid.gsites, self.grid.nd), dtype=numpy.int32)
        Nd = len(self.bs)
        ofs = [0] * Nd
        for mu in range(Nd):
            if self.ebs[mu] > 1:
                ofs[mu] = 1
        for n in range(self.nb // 2):
            sl = slice(n * self.bv, (n + 1) * self.bv)
            top = [ofs[mu] * n * self.bs[mu] for mu in range(Nd)]
            bottom = [top[mu] + self.bs[mu] for mu in range(Nd)]
            _pos = cgpt.coordinates_from_cartesian_view(
                top, bottom, self.grid.cb.cb_mask, tag, "lexicographic"
            )

            self.pos[sl, :] = _pos

    def set_BC_Ufld(self, U):
        if self.grid.nd == 4:
            self.setBC_4D(U)
        else:
            assert 0

    def setBC_4D(self, U):
        # colon = slice(None,None,None)
        # tuple([colon] * i + [ coor ] + [colon] * (nd-i))
        if self.bd[0] > 1:
            for i in range(1, self.ebs[0] + 1):
                U[0][i * self.bs[0] - 1, :, :, :] = 0
        if self.bd[1] > 1:
            for i in range(1, self.ebs[1] + 1):
                U[1][:, i * self.bs[1] - 1, :, :] = 0
        if self.bd[2] > 1:
            for i in range(1, self.ebs[2] + 1):
                U[2][:, :, i * self.bs[2] - 1, :] = 0
        if self.bd[3] > 1:
            for i in range(1, self.ebs[3] + 1):
                U[3][:, :, :, i * self.bs[3] - 1] = 0


class sap_instance:
    def __init__(self, op, bs):
        self.op = op
        self.op_blk = []
        dt = -gpt.time()

        # thanks to double copy inside operator, U only temporary
        Ublk = [sap_blk(op.U_grid, bs, eo) for eo in range(2)]
        U = [gpt.mcolor(Ublk[0].grid) for _ in range(4)]

        for eo in range(2):
            Ucoor = Ublk[eo].coor(op.U_grid)
            for mu in range(4):
                U[mu][Ublk[eo].pos] = op.U[mu][Ucoor]
            Ublk[eo].set_BC_Ufld(U)
            self.op_blk.append(op.updated(U))

        if self.op.F_grid.nd == len(bs) + 1:
            _bs = [self.op.F_grid.fdimensions[0]] + bs
        else:
            _bs = bs

        blk = [sap_blk(self.op.F_grid, _bs, eo) for eo in range(2)]
        self.pos = blk[0].pos
        self.pos.flags["WRITEABLE"] = False
        self.coor = [blk[eo].coor(op.F_grid) for eo in range(2)]

        for eo in range(2):
            self.coor[eo].flags["WRITEABLE"] = False

        dt += gpt.time()
        gpt.message(f"SAP Initialized in {dt:g} secs")


class sap:
    @params_convention(bs=None)
    def __init__(self, params):
        self.bs = params["bs"]
        assert self.bs is not None

    def __call__(self, op):
        return sap_instance(op, self.bs)


class sap_cycle:
    @params_convention(bs=None)
    def __init__(self, blk_solver, params):
        self.bs = params["bs"]
        self.blk_solver = blk_solver
        assert self.bs is not None

    def __call__(self, op):
        sap = sap_instance(op, self.bs)
        otype = sap.op.otype[0]
        src_blk = gpt.lattice(sap.op_blk[0].F_grid, otype)
        dst_blk = gpt.lattice(sap.op_blk[0].F_grid, otype)
        solver = [self.blk_solver(op) for op in sap.op_blk]
        cache = {}

        def inv(dst, src):
            dst[:] = 0
            eta = gpt.copy(src)
            ws = [gpt.copy(src) for _ in range(2)]
            cache_key_base = (
                f"{dst.describe()}_{src.describe()}_{src.grid.obj}_{dst.grid.obj}"
            )

            dt_solv = dt_distr = dt_hop = 0.0
            for eo in range(2):
                ws[0][:] = 0
                dt_distr -= gpt.time()
                cache_key = f"{cache_key_base}_{eo}_a"
                if cache_key not in cache:
                    plan = gpt.copy_plan(src_blk, eta, embed_in_communicator=eta.grid)
                    plan.destination += src_blk.view[sap.pos]
                    plan.source += eta.view[sap.coor[eo]]
                    cache[cache_key] = plan()
                cache[cache_key](src_blk, eta)
                dt_distr += gpt.time()

                dt_solv -= gpt.time()
                dst_blk[:] = 0  # for now
                solver[eo](dst_blk, src_blk)
                dt_solv += gpt.time()

                dt_distr -= gpt.time()
                cache_key = f"{cache_key_base}_{eo}_b"
                if cache_key not in cache:
                    plan = gpt.copy_plan(
                        ws[0], dst_blk, embed_in_communicator=ws[0].grid
                    )
                    plan.destination += ws[0].view[sap.coor[eo]]
                    plan.source += dst_blk.view[sap.pos]
                    cache[cache_key] = plan()
                cache[cache_key](ws[0], dst_blk)
                dt_distr += gpt.time()

                dt_hop -= gpt.time()
                if eo == 0:
                    sap.op(ws[1], ws[0])
                eta -= ws[1]
                dst += ws[0]
                dt_hop += gpt.time()

                gpt.message(
                    f"SAP cycle; |rho|^2 = {gpt.norm2(eta):g}; |dst|^2 = {gpt.norm2(dst):g}"
                )
                gpt.message(
                    f"SAP Timings: distr {dt_distr:g} secs, blk_solver {dt_solv:g} secs, hop+update {dt_hop:g} secs"
                )

        return gpt.matrix_operator(
            mat=inv,
            inv_mat=sap.op,
            adj_inv_mat=sap.op.adj(),
            adj_mat=None,  # implement adj_mat when needed
            otype=otype,
            accept_guess=(True, False),
            grid=sap.op.F_grid,
            cb=None,
        )


# sap_cycle applies K of
#
#      ( EE^-1            0     )
#  K = ( -OO^-1 OE EE-1   OO^-1 )
#
# ws0 = EE^-1 src_e
# ws1 = OE EE^-1 src_e

# eta = src - OE EE^-1 src_e
# dst = OE EE^-1 src_e
#
# ws0 = OO^-1 (src_o - OE EE^-1 src_e)
# ws1 = EO OO^-1 (src_o - OE EE^-1 src_e)
#
