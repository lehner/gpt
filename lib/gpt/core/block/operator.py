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
import gpt, sys


def operator(op, cgrid, basis):
    src_fine = gpt.lattice(basis[0])
    dst_fine = gpt.lattice(basis[0])
    verbose = gpt.default.is_verbose("block_operator")

    def mat(dst_coarse, src_coarse):
        t0 = gpt.time()
        gpt.block.promote(src_coarse, src_fine, basis)  # TODO: src/dst ordering!!!
        t1 = gpt.time()
        op(dst_fine, src_fine)
        t2 = gpt.time()
        gpt.block.project(dst_coarse, dst_fine, basis)
        t3 = gpt.time()
        if verbose:
            gpt.message(
                "Timing: %g s (promote), %g s (matrix), %g s (project)"
                % (t1 - t0, t2 - t1, t3 - t2)
            )

    otype = gpt.ot_vsinglet(len(basis))

    return gpt.matrix_operator(mat=mat, otype=otype, zero=(False, False), grid=cgrid)
