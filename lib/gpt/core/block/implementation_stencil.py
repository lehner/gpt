#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt as g


def create_stencil_operator_n_rhs(
    points, vector_parity, n_rhs, target_checkerboard, packed_right_hand_sides, args
):
    assert packed_right_hand_sides is None
    assert args == []

    # get vector type
    vector_type = None
    for p in points:
        vector_type = points[p].otype.vector_type
        grid = points[p].grid
        break
    assert vector_type is not None
    nbasis = vector_type.shape[0]
    npoints = len(points)
    lpoints = list(points.keys())
    mcoarse = [points[lpoints[ip]] for ip in range(npoints)]
    vcoarse = [g.vcomplex(grid, nbasis) for _ in range(n_rhs)]
    nbasis_blocks = len(vcoarse[0].v_obj)

    # identify zero point
    zero_point = None
    for ip in range(npoints):
        if sum([x**2 for x in lpoints[ip]]) == 0:
            zero_point = ip
    if zero_point is None:
        lpoints.append(tuple([0] * grid.nd))
        zero_point = npoints

    # create stencil
    code = []
    for i in range(n_rhs):
        ioff = nbasis_blocks * i
        for iblock in range(nbasis_blocks):
            for jblock in range(nbasis_blocks):
                matrix_index = nbasis_blocks * jblock + iblock
                for ip in range(npoints):
                    code.append(
                        (
                            iblock + ioff,  # target
                            nbasis_blocks * n_rhs + jblock + ioff,  # source
                            ip,  # source point
                            -1 if jblock == 0 and ip == 0 else iblock + ioff,  # accumulate
                            1.0,
                            [(nbasis_blocks**2 * ip + matrix_index, zero_point, 0)],
                        )
                    )
    st = g.stencil.matrix_vector(
        mcoarse[0],
        vcoarse[0],
        lpoints,
        code,
        len(code) // nbasis_blocks // n_rhs,
        vector_parity=vector_parity,
    )
    st.data_access_hints(
        list(range(n_rhs)), list(range(n_rhs, 2 * n_rhs)), list(range(len(lpoints)))
    )

    def _mat(dst, src):
        assert len(src) == n_rhs
        st(mcoarse, dst + src)
        for d in dst:
            d.checkerboard(target_checkerboard)

    return g.matrix_operator(_mat, accept_list=(True, True))
