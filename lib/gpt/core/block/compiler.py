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
import gpt.core.block.implementation_stencil as implementation_stencil
import gpt.core.block.implementation_blas as implementation_blas
import gpt.core.block.implementation_reference as implementation_reference


def adj_points(points):
    ret = {}
    for p in points:
        np = tuple([-x for x in p])
        x = g(g.adj(points[p]))
        for i in range(len(p)):
            if p[i] != 0:
                x = g.cshift(x, i, np[i])
        ret[np] = x
    return ret


def create_stencil_operator(
    points, vector_parity, target_checkerboard, implementation, packed_right_hand_sides
):
    cache = {}

    grid = None
    for p in points:
        grid = points[p].grid
        nbasis = points[p].otype.shape[0]
        otype = g.ot_vector_complex_additive_group(nbasis)
        break

    verbose = g.default.is_verbose("compiled_stencil_operator")

    vector_space = g.vector_space.explicit_grid_otype_checkerboard(grid, otype, target_checkerboard)

    if packed_right_hand_sides is not None:
        vector_space = vector_space.packed(packed_right_hand_sides)

    # select implementation
    if implementation is None:
        # default is blas
        implementation = "blas"

    # allow for arguments to be passed
    implementation, *implementation_args = implementation.split(".")
    
    # blas so far does not exist on checkerboarded grids, fall back to stencil
    if grid.cb.n == 2:
        if implementation == "blas":
            implementation = "stencil"

    # selection map
    implementation_map = {
        "stencil": implementation_stencil,
        "reference": implementation_reference,
        "blas": implementation_blas,
    }

    def delayed_matrix(get_points, tag, ip, ocb):
        def _mat(dst, src):
            n_rhs = len(src)
            key = (n_rhs, tag)
            if key not in cache:
                cache[key] = implementation_map[implementation].create_stencil_operator_n_rhs(
                    get_points(), ip, n_rhs, ocb, packed_right_hand_sides, implementation_args
                )

            if verbose:
                g.message(f"Call compiled_stencil_operator with {n_rhs} right-hand sides")
            cache[key](dst, src)

        return _mat

    cb = {0: g.even, 1: g.odd}
    _mat = delayed_matrix(lambda: points, 0, vector_parity, target_checkerboard)
    _adj_mat = delayed_matrix(
        lambda: adj_points(points), 1, target_checkerboard.tag, cb[vector_parity]
    )
    _inv_mat = None
    _adj_inv_mat = None
    if len(points) == 1:
        k = list(points.keys())[0]
        if sum([x**2 for x in k]) == 0:
            _inv_mat = delayed_matrix(
                lambda: {k: g(g.matrix.inv(points[k]))},
                2,
                target_checkerboard.tag,
                cb[vector_parity],
            )
            _adj_inv_mat = delayed_matrix(
                lambda: {k: g(g.adj(g.matrix.inv(points[k])))},
                3,
                vector_parity,
                target_checkerboard,
            )

    return g.matrix_operator(
        mat=_mat,
        inv_mat=_inv_mat,
        adj_mat=_adj_mat,
        adj_inv_mat=_adj_inv_mat,
        accept_list=(True, True),
        vector_space=vector_space,
    )


def project_points_parity(points, parity_p, parity_s):
    res = {}
    for p in points:
        # get parity of points
        if sum(p) % 2 == parity_p:
            res[p] = g.pick_checkerboard(parity_s, points[p])
    return res


def create(coarse_matrix, points, nblock):

    if nblock is None:
        nblock = 8

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
        grid_orig = points[p].grid
        break
    assert vector_type is not None
    nbasis = vector_type.shape[0]
    npoints = len(points)
    lpoints = list(points.keys())

    # algebra in double precision
    grid = grid_orig.converted(g.double)

    # create point masks
    point_masks = [g.complex(grid) for i in range(npoints)]
    ppoints = []
    L = [2] * grid.nd
    mL = [
        np.max([lpoints[i][mu] for i in range(npoints)])
        - np.min([lpoints[i][mu] for i in range(npoints)])
        for mu in range(grid.nd)
    ]
    for i in range(grid.nd):
        while L[i] < 2 * mL[i]:
            L[i] *= 2
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
        srcdag_mat_src_p = []
        g.message(f"compile: {i} / {nbasis}")
        for ip in range(0, npoints, nblock):
            i0 = ip
            i1 = min(npoints, i0 + nblock)

            cm = g.convert(
                g(
                    coarse_matrix
                    * g.expr(
                        [
                            g.convert(g(point_masks[ipi] * src_mask[i]), grid_orig.precision)
                            for ipi in range(i0, i1)
                        ]
                    )
                ),
                g.double,
            )

            for ipi in range(i0, i1):
                srcdag_mat_src_p.append(g(g.adj(point_masks[ipi]) * cm[ipi - i0]))
        t("rotate")
        g.rotate(srcdag_mat_src_p, Minv, 0, npoints, 0, npoints)
        t("accumulate")
        cache_left = {}
        for ip in range(npoints):
            points[lpoints[ip]][
                tuple([slice(None, None, None)] * (grid.nd + 1) + [i]), cache_left
            ] = g.convert(srcdag_mat_src_p[ip], grid_orig.precision)[:, cache_right]

    t()

    g.message(t)
