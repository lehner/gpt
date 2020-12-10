#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020 Tilo Wettig
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
import gpt, sys
import numpy

# need a basic container to store the group/algebra data
from gpt.core.object_type import ot_matrix_color, ot_vector_color

###
# TODO:
# su2 subgroups
# add projection to algebra and group here
#


###
# Compute structure constant
def compute_structure_constant(T, dt):
    Ndim = len(T)
    f_abc = numpy.array(
        [
            [
                [
                    numpy.trace(
                        (T[a].array @ T[b].array - T[b].array @ T[a].array) @ T[c].array
                    )
                    / numpy.trace(T[c].array @ T[c].array)
                    / 1j
                    for c in range(Ndim)
                ]
                for b in range(Ndim)
            ]
            for a in range(Ndim)
        ],
        dtype=dt,
    )

    assert numpy.abs(f_abc[0][1][2] - 1) < 1e-7

    return f_abc


###
# Convert fundamental to adjoint representation
def fundamental_to_adjoint(U_a, U_f):
    grid = U_f.grid
    T = U_f.otype.cartesian().generators(grid.precision.complex_dtype)
    V = {}
    for a in range(len(T)):
        for b in range(len(T)):
            V[a, b] = gpt.eval(2.0 * gpt.trace(T[a] * U_f * T[b] * gpt.adj(U_f)))
    gpt.merge_color(U_a, V)


###
# Base class
class ot_matrix_su_n_base(ot_matrix_color):
    def __init__(self, Nc, Ndim, name):
        self.Nc = Nc
        self.Ndim = Ndim
        super().__init__(Ndim)  # Ndim x Ndim matrix
        self.__name__ = name
        self.data_alias = lambda: ot_matrix_color(Ndim)
        self.mtab = {
            self.__name__: (lambda: self, (1, 0)),
            f"ot_vector_color({Ndim})": (lambda: ot_vector_color(Ndim), (1, 0)),
            "ot_singlet": (lambda: self, None),
        }
        self.rmtab = {
            "ot_singlet": (lambda: self, None),
        }


class ot_matrix_su_n_algebra(ot_matrix_su_n_base):
    def __init__(self, Nc, Ndim, name):
        super().__init__(Nc, Ndim, name)

    def cartesian(self):
        return self

    def coordinates(self, l, c=None):
        assert l.otype.__name__ == self.__name__
        gen = self.generators(l.grid.precision.complex_dtype)
        if c is None:
            norm = [numpy.trace(Ta.array @ Ta.array) for Ta in gen]
            return [gpt.eval(gpt.trace(l * Ta) / n) for n, Ta in zip(norm, gen)]
        else:
            l[:] = 0
            for ca, Ta in zip(c, gen):
                l += ca * Ta


class ot_matrix_su_n_group(ot_matrix_su_n_base):
    def __init__(self, Nc, Ndim, name):
        super().__init__(Nc, Ndim, name)

    def is_element(self, U):
        I = gpt.identity(U)
        err = (gpt.norm2(U * gpt.adj(U) - I) / gpt.norm2(I)) ** 0.5
        # consider additional determinant check
        return err < U.grid.precision.eps * 10.0


###
# Representations of groups
class ot_matrix_su_n_fundamental_algebra(ot_matrix_su_n_algebra):
    def __init__(self, Nc):
        super().__init__(Nc, Nc, f"ot_matrix_su_n_fundamental_algebra({Nc})")
        self.ctab = {
            f"ot_matrix_su_n_fundamental_group({Nc})": lambda dst, src: gpt.eval(
                dst, gpt.matrix.exp(src * 1j)
            )
        }

    def generators(self, dt):
        r = []

        # accumulate generators in pauli / Gell-Mann ordering
        for i in range(self.Nc):
            for j in range(i + 1, self.Nc):

                # sigma_x
                alg = numpy.zeros(shape=(self.Nc, self.Nc), dtype=dt)
                alg[i, j] = 1.0
                alg[j, i] = 1.0
                r.append(alg)

                # sigma_y
                alg = numpy.zeros(shape=(self.Nc, self.Nc), dtype=dt)
                alg[i, j] = -1j
                alg[j, i] = 1j
                r.append(alg)

                # sigma_z
                if j == i + 1:
                    alg = numpy.zeros(shape=(self.Nc, self.Nc), dtype=dt)
                    for l in range(j):
                        alg[l, l] = 1.0
                    alg[j, j] = -j
                    r.append(alg)

        # need to satisfy normalization Tr(T_a T_b) = 1/2 delta_{ab}
        for alg in r:
            alg /= (numpy.trace(numpy.dot(alg, alg)) * 2.0) ** 0.5

        # return gpt_object version
        algebra_otype = ot_matrix_su_n_fundamental_algebra(self.Nc)
        return [gpt.gpt_object(i, algebra_otype) for i in r]


class ot_matrix_su_n_fundamental_group(ot_matrix_su_n_group):
    cache = {}

    def __init__(self, Nc):
        super().__init__(Nc, Nc, f"ot_matrix_su_n_fundamental_group({Nc})")
        self.ctab = {
            f"ot_matrix_su_n_adjoint_group({Nc})": fundamental_to_adjoint,
            f"ot_matrix_su_n_fundamental_algebra({Nc})": lambda dst, src: gpt.eval(
                dst, gpt.matrix.log(src) / 1j
            ),
        }

    def cartesian(self):
        return ot_matrix_su_n_fundamental_algebra(self.Nc)

    def su_2_subgroups(self):
        N = (self.Nc * (self.Nc - 1)) // 2
        r = []
        for i in range(N - 1):
            for j in range(i + 1, N):
                r.append((i, j))
        assert len(r) == N
        return r

    def su_2_extract(self, u2, U, idx):
        assert u2.otype.Nc == 2 and u2.otype.Ndim == 2
        idx = list(idx)
        cache = ot_matrix_su_n_fundamental_group.cache
        cache_key = f"{self.Nc}_{idx}"
        if cache_key not in cache:
            plan = gpt.copy_plan(u2, U)
            for i in range(2):
                for j in range(2):
                    plan.destination += u2.view[:, i, j]
                    plan.source += U.view[:, idx[i], idx[j]]
            cache[cache_key] = plan()
        cache[cache_key](u2, U)
        u2 @= u2 - gpt.adj(u2) + gpt.identity(u2) * gpt.trace(gpt.adj(u2))
        pass

    def su_2_insert(self, U, u2, idx):
        pass


class ot_matrix_su_n_adjoint_algebra(ot_matrix_su_n_algebra):
    f = {}

    def __init__(self, Nc):
        super().__init__(Nc, Nc * Nc - 1, f"ot_matrix_su_n_adjoint_algebra({Nc})")
        self.ctab[f"ot_matrix_su_n_adjoint_group({Nc})"] = lambda dst, src: gpt.eval(
            dst, gpt.matrix.exp(src * 1j)
        )

    def generators(self, dt):
        T_f = ot_matrix_su_n_fundamental_algebra(self.Nc).generators(dt)
        if self.Nc not in ot_matrix_su_n_adjoint_algebra.f:
            ot_matrix_su_n_adjoint_algebra.f[self.Nc] = compute_structure_constant(
                T_f, dt
            )
            # assert compute_structure_constant(ot_matrix_su_n_fundamental_algebra(2).generators(dt),dt)[0][1][2] == 1
            # assert compute_structure_constant(ot_matrix_su_n_adjoint_algebra(2).generators(dt),dt)[0][1][2] == 1

        r = []
        for a in range(self.Ndim):
            r.append(ot_matrix_su_n_adjoint_algebra.f[self.Nc][a] / 1j)

        # return gpt_object version
        algebra_otype = ot_matrix_su_n_adjoint_algebra(self.Nc)
        return [gpt.gpt_object(i, algebra_otype) for i in r]


class ot_matrix_su_n_adjoint_group(ot_matrix_su_n_group):
    def __init__(self, Nc):
        super().__init__(Nc, Nc * Nc - 1, f"ot_matrix_su_n_adjoint_group({Nc})")
        self.ctab = {
            f"ot_matrix_su_n_adjoint_algebra({Nc})": lambda dst, src: gpt.eval(
                dst, gpt.matrix.log(src) / 1j
            )
        }

    def cartesian(self):
        return ot_matrix_su_n_adjoint_algebra(self.Nc)
