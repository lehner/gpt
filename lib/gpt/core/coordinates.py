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
import gpt, cgpt, numpy


class local_coordinates(numpy.ndarray):
    pass


def coordinates(o, order="lexicographic"):
    if type(o) == gpt.grid and o.cb.n == 1:
        return coordinates((o, gpt.none), order=order)
    elif type(o) == tuple and type(o[0]) == gpt.grid and len(o) == 2:
        dim = len(o[0].ldimensions)
        cb = o[1].tag
        checker_dim_mask = o[0].cb.cb_mask
        cbf = [o[0].fdimensions[i] // o[0].gdimensions[i] for i in range(dim)]
        top = [
            o[0].processor_coor[i] * o[0].ldimensions[i] * cbf[i] for i in range(dim)
        ]
        bottom = [top[i] + o[0].ldimensions[i] * cbf[i] for i in range(dim)]
        return cgpt.coordinates_from_cartesian_view(
            top, bottom, checker_dim_mask, cb, order
        ).view(local_coordinates)
    elif type(o) == gpt.lattice:
        return coordinates((o.grid, o.checkerboard()), order=order)
    elif type(o) == gpt.cartesian_view:
        return cgpt.coordinates_from_cartesian_view(
            o.top, o.bottom, o.checker_dim_mask, o.cb, order
        )
    else:
        assert 0


def apply_exp_ixp(dst, src, p):
    # TODO: add sparse field support (x.internal_coordinates(), x.coordinates())
    x = gpt.coordinates(src)

    # create phase field
    phase = gpt.complex(src.grid)
    phase.checkerboard(src.checkerboard())
    phase[x] = cgpt.coordinates_momentum_phase(x, p, src.grid.precision)
    dst @= phase * src


def exp_ixp(p):

    if type(p) == list:
        return [exp_ixp(x) for x in p]
    elif isinstance(p, numpy.ndarray):
        p = p.tolist()

    def mat(dst, src):
        return apply_exp_ixp(dst, src, p)

    def inv_mat(dst, src):
        return apply_exp_ixp(dst, src, [-x for x in p])

    # do not specify grid or otype, i.e., accept all
    return gpt.matrix_operator(
        mat=mat, adj_mat=inv_mat, inv_mat=inv_mat, adj_inv_mat=mat
    )


def fft(dims=None):
    def mat(dst, src, sign):
        d = dims if dims is not None else list(range(src.grid.nd))
        assert dst.otype.__name__ == src.otype.__name__
        for i in dst.otype.v_idx:
            cgpt.fft(dst.v_obj[i], src.v_obj[i], d, sign)

    def mat_forward(dst, src):
        mat(dst, src, 1)

    def mat_backward(dst, src):
        mat(dst, src, -1)

    return gpt.matrix_operator(
        mat=mat_forward,
        adj_mat=mat_backward,
        inv_mat=mat_backward,
        adj_inv_mat=mat_forward,
    )


def coordinate_mask(field, mask):
    assert isinstance(mask, numpy.ndarray)
    assert field.otype == gpt.ot_singlet

    x = gpt.coordinates(field)
    field[x] = mask.astype(field.grid.precision.complex_dtype).reshape((len(mask), 1))
