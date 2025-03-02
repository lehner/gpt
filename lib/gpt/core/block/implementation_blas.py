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


def create_stencil_operator_n_rhs(points, n_rhs):

    # get vector type
    vector_type = None
    for p in points:
        vector_type = points[p].otype.vector_type
        grid = points[p].grid
        break
    assert vector_type is not None
    nbasis = vector_type.shape[0]
    npoints = len(points)
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
        for j in range(nd):
            margin[j] = max(margin[j], abs(lpoints[ip][j]))

    # create packs
    pM = [g.pack(m) for m in mcoarse]
    pV = g.pack(vcoarse)

    # create buffers
    bM = [m.to_accelerator_buffer().merged_axes(-3, -2) for m in pM]
    bR = pV.to_accelerator_buffer(margin=margin)
    bL = pV.to_accelerator_buffer()
    idxL = bL.indices(range(nd))

    halo_exchange = bR.halo_exchange(grid, margin=margin, max_point_sqr=max_point_sqr)

    cR = bR.coordinates(range(nd))
    bulkR = bR.bulk(cR, margin=margin)

    blas = g.blas()
    for ip in range(npoints):
        idxR = bR.indices(range(nd), shift=lpoints[ip])[bulkR]
        blas.gemm(1.0, bM[ip][idxL], bR[idxR].T, 0.0 if ip == 0 else 1.0, bL[idxL].T)

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
        pL.from_accelerator_buffer(bL)
        t()

        if verbose:
            g.message(t)

    return g.matrix_operator(_mat, accept_list=(True, True))
