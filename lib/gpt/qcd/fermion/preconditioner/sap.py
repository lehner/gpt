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

import gpt, cgpt, numpy

# ix = x[0] + lat[0]*(x[1] + lat[2]*...)
def index2coor(ix, lat):
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
            raise SapError(f"Sap Blocks should define an even/odd grid")

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

    def coor(self, grid, tag=None):
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


class sap:
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