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
import gpt, cgpt, numpy

def coordinates(o, order = "grid"):
    if type(o) == gpt.grid and o.cb.n == 1:
        dim=len(o.ldimensions)
        top=[ o.processor_coor[i]*o.ldimensions[i] for i in range(dim) ]
        bottom=[ top[i] + o.ldimensions[i] for i in range(dim) ]
        checker_dim_mask=[ 0 ] * dim
        return cgpt.coordinates_from_cartesian_view(top,bottom,checker_dim_mask,None,order)
    elif type(o) == tuple and type(o[0]) == gpt.grid and len(o) == 2:
        dim=len(o[0].ldimensions)
        cb=o[1].tag
        checker_dim_mask=o[0].cb.cb_mask
        cbf=[ o[0].fdimensions[i] // o[0].gdimensions[i] for i in range(dim) ]
        top=[ o[0].processor_coor[i]*o[0].ldimensions[i]*cbf[i] for i in range(dim) ]
        bottom=[ top[i] + o[0].ldimensions[i]*cbf[i] for i in range(dim) ]
        return cgpt.coordinates_from_cartesian_view(top,bottom,checker_dim_mask,cb,order)
    elif type(o) == gpt.lattice:
        return coordinates( (o.grid,o.checkerboard()), order = order )
    elif type(o) == gpt.cartesian_view:
        return cgpt.coordinates_from_cartesian_view(o.top,o.bottom,o.checker_dim_mask,o.cb,order)
    else:
        assert(0)

def apply_exp_ixp(dst,src,p):
    # TODO: add sparse field support (x.internal_coordinates(), x.coordinates())
    x=src.mview_coordinates()

    # create phase field
    phase=gpt.complex(src.grid)
    phase.checkerboard(src.checkerboard())
    phase[x]=cgpt.coordinates_momentum_phase(x,p,src.grid.precision)
    dst @= phase * src

def exp_ixp(p):

    if type(p) == list:
        return [ momentum_phase(x) for x in p ]
    elif type(p) == numpy.ndarray:
        p=p.tolist()

    mat=lambda dst,src: apply_exp_ixp(dst,src,p)
    inv_mat=lambda dst,src: apply_exp_ixp(dst,src,[ -x for x in p ])

    # do not specify grid or otype, i.e., accept all
    return gpt.matrix_operator(mat = mat,
                               adj_mat = inv_mat,
                               inv_mat = inv_mat,
                               adj_inv_mat = mat)
