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
import gpt,sys

class operator_on_fine:
    def __init__(self, op, cgrid, basis):
        self.op = op
        self.src_fine=gpt.lattice(basis[0])
        self.dst_fine=gpt.lattice(basis[0])
        self.basis=basis
        self.verbose=gpt.default.is_verbose("block_operator")

    def __call__(self, dst_coarse, src_coarse):
        tpf=gpt.time()
        gpt.prefetch([src_coarse,dst_coarse,self.src_fine,self.dst_fine,self.basis],gpt.to_accelerator)
        t0=gpt.time()
        gpt.block.promote(src_coarse,self.src_fine,self.basis) # TODO: src/dst ordering!!!
        t1=gpt.time()
        self.op(self.dst_fine,self.src_fine)
        t2=gpt.time()
        gpt.block.project(dst_coarse,self.dst_fine,self.basis)
        t3=gpt.time()
        if self.verbose:
            gpt.message("Timing: %g s (promote), %g s (matrix), %g s (project), %g s (prefetch)" % (t1-t0,t2-t1,t3-t2,t0-tpf))

def operator(op, cgrid, basis):
    # If possible, directly implement op
    return operator_on_fine(op,cgrid,basis)
