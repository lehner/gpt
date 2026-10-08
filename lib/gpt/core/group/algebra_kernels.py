#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


class _algebra_kernels:
    # site-local tensor kernels between algebra fields (N x N) and
    # adjoint-algebra matrices (ng x ng), with the generators T_b as constant
    # fields (read as kernel temporaries, as in local_stout):
    #   rows(M, l)    M[a, b] = tr(l_a T_b) / tr(T_b T_b)   (coordinates of l_a)
    #   combine(r, K) r_a     = -sum_b K[a, b] T_b
    # one kernel each instead of ng^2 (coordinates) or ng^2 (products) kernels
    def __init__(self, grid, otype_cartesian):
        N = otype_cartesian.shape[0]
        generators = otype_cartesian.generators(grid.precision.complex_dtype)
        ng = len(generators)
        norm = [complex(np.trace(t.array @ t.array)) for t in generators]
        # (the scale is applied once per element, so all norms must agree)
        assert all(abs(n - norm[0]) < 10 * grid.precision.eps for n in norm)
        ti = g.stencil.tensor_instructions
        self.N, self.ng = N, ng
        self.norm = complex(norm[0]).real
        self.field_generators = [g.lattice(grid, otype_cartesian) for _ in range(ng)]
        for f, t in zip(self.field_generators, generators):
            f[:] = t
        nonzero = [
            [(i, j) for i in range(N) for j in range(N) if abs(t.array[i, j]) != 0.0]
            for t in generators
        ]

        # fields: M, l_0..l_{ng-1}, T_0..T_{ng-1}
        code = []
        for a in range(ng):
            for b in range(ng):
                for k, (j, i) in enumerate(nonzero[b]):
                    code.append(
                        (
                            0,
                            a * ng + b,
                            ti.mov if k == 0 else ti.inc,
                            1.0,
                            [(1 + a, 0, i * N + j), (-(1 + ng + b), 0, j * N + i)],
                        )
                    )
                code.append((0, a * ng + b, ti.mul, 1.0 / norm[0], [(0, 0, a * ng + b)]))
        M = g.lattice(grid, g.ot_matrix_su_n_adjoint_algebra(N))
        self._rows = g.stencil.tensor(M, [(0,) * grid.nd], code, [(len(code), 1)])

        # fields: r_0..r_{ng-1}, K, T_0..T_{ng-1}
        code = []
        for a in range(ng):
            first = {}
            for b in range(ng):
                for i, j in nonzero[b]:
                    code.append(
                        (
                            a,
                            i * N + j,
                            ti.inc if (i, j) in first else ti.mov,
                            1.0,
                            [(ng, 0, a * ng + b), (-(ng + 1 + b), 0, i * N + j)],
                        )
                    )
                    first[i, j] = True
            # (every element is covered by some generator)
            assert len(first) == N * N
            for e in range(N * N):
                code.append((a, e, ti.mul, -1.0, [(a, 0, e)]))
        self._combine = g.stencil.tensor(
            self.field_generators[0], [(0,) * grid.nd], code, [(len(code), 1)]
        )

    def rows(self, M, l):
        self._rows(M, *l, *self.field_generators)
        return M

    def combine(self, r, K):
        self._combine(*r, K, *self.field_generators)
        return r


def algebra_kernels(grid, otype_cartesian):
    """Site-local tensor kernels between the algebra fields of
    otype_cartesian (an SU(N) algebra, N x N) and adjoint-algebra matrices
    (ng x ng), one set per (grid, algebra type), kept on the grid:

      rows(M, l)    M[a, b] = tr(l_a T_b) / tr(T_b T_b)  (coordinates of l_a)
      combine(r, K) r_a     = -sum_b K[a, b] T_b

    with the generators T_b as constant fields (field_generators), their common
    norm tr(T_b T_b) (norm) and number (ng)."""
    cache = grid.__dict__.setdefault("_algebra_kernels", {})
    key = otype_cartesian.__name__
    if key not in cache:
        cache[key] = _algebra_kernels(grid, otype_cartesian)
    return cache[key]
