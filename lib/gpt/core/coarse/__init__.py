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
import gpt, cgpt, sys

def create_links(A, fmat, basis):
    # NOTE: we expect the blocks in the basis vectors
    # to already be orthogonalized!

    # directions/displacements we coarsen for
    dirs = [0, 1, 2, 3]  # TODO: for 5d, this needs += 1
    disp = +1
    selflink = 8
    hermitian = True  # for now

    # setup fields
    Mvr = [gpt.lattice(basis[0]) for i in range(9)]  # (needed by current grid)
    Mvre, Mvro, tmp = gpt.lattice(basis[0]), gpt.lattice(basis[0]), gpt.lattice(basis[0]),
    oproj = gpt.complex(A[0].grid)
    selfproj = gpt.vcomplex(A[0].grid, len(basis))

    tmp2 = gpt.lattice(basis[0])

    # setup masks
    onemask, evenmask, oddmask = gpt.complex(basis[0].grid), gpt.complex(basis[0].grid), gpt.complex(basis[0].grid)
    dirmasks = [gpt.complex(basis[0].grid) for d in dirs]

    # fill even/odd masks using temporary mask on eo grid
    onemask[:] = 1.
    fgrid = basis[0].grid
    fgrid_eo = gpt.grid(fgrid.fdimensions, fgrid.precision, gpt.redblack)
    evenmask_eo = gpt.lattice(fgrid_eo, evenmask.otype)
    evenmask_eo.checkerboard(gpt.even)
    evenmask_eo[:] = 1.
    gpt.set_cb(evenmask, evenmask_eo)
    oddmask @= onemask - evenmask

    # fill directional masks
    for d in dirs:
        dirmasks[d][:] = 1.

    for d in dirs:
        # latticecoordinate for current dir
        # use coor to create masks for even, odd, directions
        # coor = basis[0].mview_coordinates()
        # gpt.message("coor = ", coor)
        pass

    for i, vr in enumerate(basis):
        # gpt.message("i, vr = ", i, vr)
        for d in dirs:  # this triggers four comms -> need to expose DhopdirAll from Grid but problem with vector<Lattice<... in rhs
            fmat.Mdir(Mvr[d], vr, d, disp)
            # NOTE: this works

        # coarsen directional terms
        for d in dirs:
            dirmasks[d][:] = 1.
            for j, vl in enumerate(basis): # can combine this together into 1 routine
                oproj2_b = gpt.norm2(oproj)
                gpt.block.maskedInnerProduct(oproj, oddmask, vl, Mvr[d]) # NOTE: oddmask only for testing, will be dirmask
                oproj2_a = gpt.norm2(oproj)
                assert(oproj2_a != oproj2_b)
                gpt.message("oproj before/after = %e/%e" % (oproj2_b, oproj2_a))
                # gpt.block.maskedInnerProduct(oproj, dirmasks[d], vl, Mvr[d])
                # TODO: write oproj to link
                # gpt.message("i, d, j, oproj = ", i, d, j, oproj)

        # apply diagonal term for both cbs separately
        tmp @= evenmask * (fmat.M * (vr * evenmask)) + oddmask * (fmat.M * (vr * oddmask))
        tmp2 @= evenmask * fmat.M * vr * evenmask + oddmask * fmat.M * vr * oddmask
        diff2 = gpt.norm2(tmp2 - tmp)
        assert(diff2 == 0.)
        gpt.message("diff = ", diff2)
         # NOTE: this works

        # gpt.message("i, tmp = ", i, tmp)

        # coarsen diagonal term
        gpt.block.project(selfproj, tmp, basis)
        # TODO: write selfproj to link (outer product with slice of existing one?)
        # A[selflink] = ...
        # gpt.message("i, selfproj = ", i, selfproj)

    # communicate opposite links
    for d in dirs:
        dd = d + len(dirs)
        shift_disp = disp * -1
        if hermitian:
            A[dd] @= gpt.adj(gpt.cshift(A[d], d, shift_disp))
        else:
            # linktmp = ... # TODO internal index manipulation for coarse spin dofs
            A[dd] @= gpt.adj(gpt.cshift(linktmp, d, shift_disp))

    rng = gpt.random("coarse_links")
    [gpt.message("A2[%d] = %e" % (i, gpt.norm2(A[i]))) for i in range(9)]
    rng.cnormal(A)
    [gpt.message("A2[%d] = %e" % (i, gpt.norm2(A[i]))) for i in range(9)]

def recreate_links(A, fmat, basis):
    create_links(A, fmat, basis)
