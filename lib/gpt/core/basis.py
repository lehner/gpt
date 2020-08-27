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
import numpy


def orthogonalize(w, basis, ips=None, nblock=4):
    # verbosity
    verbose = gpt.default.is_verbose("orthogonalize")
    n = len(basis)
    if n == 0:
        return
    grid = basis[0].grid
    i = 0
    t_rank_inner_product = 0.0
    t_globalSum = 0.0
    t_linearCombination = 0.0
    for i in range(0, n, nblock):
        t_rank_inner_product -= gpt.time()
        lip = gpt.rank_inner_product(basis[i : i + nblock], w)
        t_rank_inner_product += gpt.time()
        t_globalSum -= gpt.time()
        grid.globalsum(lip)
        lip = [complex(x) for x in lip]
        t_globalSum += gpt.time()
        if ips is not None:
            for j in range(len(lip)):
                ips[i + j] = lip[j]
        expr = w - lip[0] * basis[i + 0]
        for j in range(1, len(lip)):
            expr -= lip[j] * basis[i + j]
        t_linearCombination -= gpt.time()
        w @= expr
        t_linearCombination += gpt.time()
    if verbose:
        gpt.message(
            "Timing Ortho: %g rank_inner_product, %g globalsum, %g lc"
            % (t_rank_inner_product, t_globalSum, t_linearCombination)
        )


def linear_combination(r, basis, Qt):
    assert len(basis[0].v_obj) == len(r.v_obj)
    Qt = numpy.array(Qt, dtype=numpy.complex128)
    for i in r.otype.v_idx:
        cgpt.linear_combination(r.v_obj[i], basis, Qt, i)


def rotate(basis, Qt, j0, j1, k0, k1):
    for i in basis[0].otype.v_idx:
        cgpt.rotate(basis, Qt, j0, j1, k0, k1, i)


def qr_decomposition(lmd, lme, Nk, Nm, Qt, Dsh, kmin, kmax):
    return cgpt.qr_decomposition(lmd, lme, Nk, Nm, Qt, Dsh, kmin, kmax)


def g5c(src):
    if hasattr(src.otype, "fundamental"):
        nbasis = src.otype.shape[0]
        assert nbasis % 2 == 0
        nb = nbasis // 2
        return gpt.vcomplex([1] * nb + [-1] * nb, nbasis)
    else:
        return gpt.gamma[5]


def split_chiral(basis, factor=None):
    nbasis = len(basis)
    assert nbasis % 2 == 0
    nb = nbasis // 2
    factor = 0.5 if factor is None else factor
    g5 = g5c(basis[0])
    tmp = gpt.lattice(basis[0])
    for n in range(nb):
        tmp @= g5 * basis[n]
        basis[n + nb] @= (basis[n] - tmp) * factor
        basis[n] @= (basis[n] + tmp) * factor


def unsplit_chiral(basis, factor=None):
    nbasis = len(basis)
    assert nbasis % 2 == 0
    nb = nbasis // 2
    factor = 0.5 if factor is None else factor
    rev_factor = 0.5 / factor
    for n in range(nb):
        basis[n] @= (basis[n] + basis[n + nb]) * rev_factor
