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


def create_stencil_operator_n_rhs(points, ip, n_rhs, ocb, packed_right_hand_sides, args):

    # get vector type
    vector_type = None
    for p in points:
        vector_type = points[p].otype.vector_type
        grid = points[p].grid
        break

    dim_offset = 0
    grid_reduced = grid
    if packed_right_hand_sides is not None:
        grid = grid.inserted_dimension(0, packed_right_hand_sides)
        dim_offset = 1
        assert n_rhs == 1

    assert vector_type is not None
    nbasis = vector_type.shape[0]
    npoints = len(points)

    # parse arguments
    if len(args) > 0:
        nparallel = int(args[0])
        assert npoints % nparallel == 0
    else:
        nparallel = 1

    if len(args) > 1:
        compute_precision = args[1]
    else:
        compute_precision = None

    npoints_parallel = npoints // nparallel
    
    lpoints_orig = list(points.keys())
    lpoints = [tuple(reversed(list(x))) for x in lpoints_orig]
    mcoarse = [points[lpoints_orig[ip]] for ip in range(npoints)]
    vcoarse = [g.vcomplex(grid, nbasis) for _ in range(n_rhs)]
    verbose = g.default.is_verbose("coarse_performance")

    # read off margin and max_point_sqr
    nd = grid.nd
    max_point_sqr = 0
    margin = [0] * nd

    for ip in range(npoints):
        point_sqr = sum([x**2 for x in lpoints[ip]])
        max_point_sqr = max(point_sqr, max_point_sqr)
        for j in range(nd - dim_offset):
            margin[j] = max(margin[j], abs(lpoints[ip][j]))

    margin_reduced = margin if packed_right_hand_sides is None else margin[:-1]

    # create packs
    pM = [g.pack(m) for m in mcoarse]
    pV = g.pack(vcoarse)

    # create buffers
    bM = [m.to_accelerator_buffer().merged_axes(-3, -2) for m in pM]
    bR = pV.to_accelerator_buffer(margin=margin)
    bL = [pV.to_accelerator_buffer() for _ in range(nparallel)]

    if packed_right_hand_sides is not None:
        bR = bR.merged_axes(-3, -2)
        for b in range(nparallel):
            bL[b] = bL[b].merged_axes(-3, -2)

    idxL = bL[0].indices(range(nd - dim_offset))

    halo_exchange = bR.halo_exchange(
        grid_reduced, margin=margin_reduced, max_point_sqr=max_point_sqr
    )

    cR = bR.coordinates(range(nd - dim_offset))
    bulkR = bR.bulk(cR, margin=margin_reduced)

    blas = g.blas()
    for a in range(npoints_parallel):
        A = []
        B = []
        C = []
        for b in range(nparallel):
            ip = a * nparallel + b
            idxR = bR.indices(range(nd - dim_offset), shift=lpoints[ip])[bulkR]
            A.append(bM[ip][idxL])
            B.append(bR[idxR].T)
            C.append(bL[b][idxL].T)
        blas.gemm(1.0, A, B, 0.0 if a == 0 else 1.0, C, precision=compute_precision)

    if nparallel > 1:
        blas.accumulate(bL)

    def _mat(dst, src):
        assert len(src) == n_rhs

        t = g.timer("coarse")
        t("pack")
        pR = g.pack(src, fast=True)
        pL = g.pack(dst, fast=True)
        t("import")
        pR.to_accelerator_buffer(target_buffer=bR, margin=margin)
        t("halo exchange")
        halo_exchange()
        t("gemm")
        blas()
        t("export")
        pL.from_accelerator_buffer(bL[0])
        t()

        if verbose:
            g.message(t)

    return g.matrix_operator(_mat, accept_list=(True, True))
