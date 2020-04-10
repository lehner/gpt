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
import cgpt

def project(coarse, fine, basis):
    for i in coarse.idx:
        cgpt.block_project(coarse.v[i].obj,fine.obj,basis[coarse.n0[i]:coarse.n1[i]])

def promote(coarse, fine, basis):
    fine[:]=0
    tmp=gpt.lattice(fine)
    for i in coarse.idx:
        cgpt.block_promote(coarse.v[i].obj,tmp.obj,basis[coarse.n0[i]:coarse.n1[i]])
        fine += tmp

def orthogonalize(coarse_grid, basis):
    assert(type(coarse_grid) == gpt.grid)
    coarse_tmp=gpt.complex(coarse_grid)
    cgpt.block_orthogonalize(coarse_tmp.obj,basis)




