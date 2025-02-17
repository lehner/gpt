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
import numpy as np
from gpt.params import params_convention


def reference_operator(points):
    def _mat(dst, src):
        for i in range(len(src)):
            dst[i][:] = 0
            for p in points:
                src_i = src[i]
                for mu in range(len(p)):
                    if p[mu] != 0:
                        src_i = g.cshift(src_i, mu, p[mu])
                assert dst[i].checkerboard() == src_i.checkerboard()
                assert points[p].checkerboard() == src_i.checkerboard()
                dst[i] += points[p] * src_i

    return g.matrix_operator(_mat, accept_list=(True, True))


def create_stencil_operator_n_rhs(points, vector_parity, n_rhs, target_checkerboard):

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


def create_stencil_operator(points, vector_parity, target_checkerboard):
    cache = {}

    def _mat(dst, src):
        n_rhs = len(src)
        if n_rhs not in cache:
            cache[n_rhs] = create_stencil_operator_n_rhs(
                points, vector_parity, n_rhs, target_checkerboard
            )

        cache[n_rhs](dst, src)

    return g.matrix_operator(_mat, accept_list=(True, True))


def project_points_parity(points, parity_p, parity_s):
    res = {}
    for p in points:
        # get parity of points
        if sum(p) % 2 == parity_p:
            res[p] = g.pick_checkerboard(parity_s, points[p])
    return res


def create(coarse_matrix, points):

    # src_i = exp(i x.point_i 2pi/l)
    # dst_i = matrix_j exp(i (x+point_j).point_i 2pi/l)
    # src_i.dst_i = matrix_j exp(i (x+point_j).point_i 2pi/l) exp(-i x.point_i 2pi/l)
    #            = matrix_j exp(i point_j.point_i 2pi/l)
    #            = M[i,j] matrix_j   ->  matrix = M^{-1} src_i.dst_i

    # timer
    t = g.timer("coarsen_performance")
    t("misc")

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

    # create point masks
    point_masks = [g.complex(grid) for i in range(npoints)]
    ppoints = []
    L = grid.fdimensions
    for i in range(npoints):
        l = point_masks[i]
        l[:] = 1.0
        ppoints.append(np.array(lpoints[i], dtype=np.int32) * 2.0 * np.pi / L)
        l @= g.exp_ixp(ppoints[i]) * l

    # create position mask in fourier space
    src_mask = []
    for i in range(nbasis):
        src = g.lattice(grid, vector_type)
        te = g.tensor(
            np.array([1.0 if i == j else 0.0 for j in range(nbasis)], dtype=np.complex128),
            vector_type,
        )
        src[:] = te
        src_mask.append(src)

    # create momentum mask
    M = np.zeros(shape=(npoints, npoints), dtype=np.complex128)
    for i in range(npoints):
        for j in range(npoints):
            M[i, j] = np.exp(1j * np.dot(ppoints[i], lpoints[j]))

    # det = eps_{mu0,mu1,mu2,...} e^{1j*v[0]*w[mu0] + 1j*v[1]*w[mu1] + ...}

    t("inverse M")
    det = abs(np.linalg.det(M))
    assert det > 1e-10
    Minv = np.linalg.inv(M)

    # different complex phases per point
    cache_right = {}
    for i in range(nbasis):
        t("apply coarse")
        srcdag_mat_src_p = [
            g(g.adj(point_masks[ip]) * coarse_matrix * (point_masks[ip] * src_mask[i]))
            for ip in range(npoints)
        ]
        t("rotate")
        g.rotate(srcdag_mat_src_p, Minv, 0, npoints, 0, npoints)
        t("accumulate")
        cache_left = {}
        for ip in range(npoints):
            points[lpoints[ip]][
                tuple([slice(None, None, None)] * (grid.nd + 1) + [i]), cache_left
            ] = srcdag_mat_src_p[ip][:, cache_right]

    t()

    g.message(t)
