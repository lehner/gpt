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
import gpt, sys
from gpt.core.expr import factor
from gpt.core.vector_space import implicit


fingerprint = gpt.default.get_int("--fingerprint", 0) > 2


def make_list(accept_list):
    return True if accept_list is False else accept_list


#
# A^dag (A^-1)^dag = (A^-1 A)^dag = 1^\dag = 1
# (A^dag)^-1 = (A^-1)^dag
#
class matrix_operator(factor):
    #
    # lhs = A rhs
    # vector_space = (lhs.vector_space,rhs.vector_space)
    # accept_guess = (accept_guess_for_mat,accept_guess_for_inv_mat)
    #
    # accept_list:
    #  False    : lhs, rhs are lattice objects
    #  True     : lhs, rhs are lists of lattice objects with len(lhs) == len(rhs)
    #  callable : lhs, rhs are lists of lattice objects with len(lhs) == callable(rhs)
    def __init__(
        self,
        mat,
        adj_mat=None,
        inv_mat=None,
        adj_inv_mat=None,
        vector_space=None,
        accept_guess=(False, False),
        accept_list=False,
    ):
        self.inheritance = None
        self.mat = mat
        self.adj_mat = adj_mat
        self.inv_mat = inv_mat
        self.adj_inv_mat = adj_inv_mat
        self.accept_list = accept_list
        self.lhs_length = (lambda rhs: len(rhs)) if not callable(accept_list) else accept_list

        if fingerprint:

            def call_matrix_operator(dst, src):
                l = gpt.fingerprint.log()
                l("source", src)
                mat(dst, src)
                l("destination", dst)
                l()

            self.mat = call_matrix_operator

        # this allows for automatic application of tensor versions
        # also should handle lists of lattices
        if vector_space is None:
            vector_space = implicit()

        self.vector_space = (
            vector_space if isinstance(vector_space, tuple) else (vector_space, vector_space)
        )

        # do we request, e.g., the lhs of lhs = A rhs to be initialized to zero
        # if it is not given?
        self.accept_guess = (
            accept_guess if isinstance(accept_guess, tuple) else (accept_guess, accept_guess)
        )

    def specialized_singlet_callable(self):
        # removing possible overhead for specialized call
        return self.mat if not self.accept_list else self

    def specialized_list_callable(self):
        # removing possible overhead for specialized call
        return self.mat if self.accept_list else self

    def inv(self):
        return matrix_operator(
            mat=self.inv_mat,
            adj_mat=self.adj_inv_mat,
            inv_mat=self.mat,
            adj_inv_mat=self.adj_mat,
            vector_space=tuple(reversed(self.vector_space)),
            accept_guess=tuple(reversed(self.accept_guess)),
            accept_list=self.accept_list,
        )

    def adj(self):
        return matrix_operator(
            mat=self.adj_mat,
            adj_mat=self.mat,
            inv_mat=self.adj_inv_mat,
            adj_inv_mat=self.inv_mat,
            vector_space=tuple(reversed(self.vector_space)),
            accept_guess=tuple(reversed(self.accept_guess)),
            accept_list=self.accept_list,
        )

    def __mul__(self, other):
        if isinstance(other, matrix_operator):
            return matrix_operator_product([self, other])
        else:
            return gpt.expr(other).__rmul__(self)

    def __rmul__(self, other):
        return gpt.expr(other).__mul__(self)

    def clone(self):
        return matrix_operator(
            mat=self.mat,
            adj_mat=self.adj_mat,
            inv_mat=self.inv_mat,
            adj_inv_mat=self.adj_inv_mat,
            vector_space=(self.vector_space[0].clone(), self.vector_space[1].clone()),
            accept_guess=self.accept_guess,
            accept_list=self.accept_list,
        )

    def converted(self, to_precision, timing_wrapper=None):
        assert all([d is not None for d in self.vector_space])

        vector_space = tuple([d.converted(to_precision) for d in self.vector_space])
        accept_guess = self.accept_guess

        def _converted(dst, src, mat, l, r, t=lambda x=None: None):
            t("converted: setup")

            conv_src = [self.vector_space[r].lattice(None, x.otype, x.checkerboard()) for x in src]
            conv_dst = [self.vector_space[l].lattice(None, x.otype, x.checkerboard()) for x in dst]

            t("converted: convert")

            gpt.convert(conv_src, src)
            if accept_guess[l]:
                gpt.convert(conv_dst, dst)

            t("converted: matrix")

            mat(conv_dst, conv_src)

            t("converted: convert")

            gpt.convert(dst, conv_dst)

            t()

        if timing_wrapper is not None:
            _converted = timing_wrapper(_converted)

        return matrix_operator(
            mat=lambda dst, src: _converted(dst, src, self, 0, 1),
            adj_mat=lambda dst, src: _converted(dst, src, self.adj(), 1, 0),
            inv_mat=lambda dst, src: _converted(dst, src, self.inv(), 1, 0),
            adj_inv_mat=lambda dst, src: _converted(dst, src, self.adj().inv(), 0, 1),
            vector_space=vector_space,
            accept_guess=accept_guess,
            accept_list=make_list(self.accept_list),
        )

    def grouped(self, max_group_size):
        def _grouped(dst, src, mat):
            n = len(src)
            r = self.lhs_length(src) // n
            for i in range(0, n, max_group_size):
                mat(
                    [dst[l * n + i + j] for l in range(r) for j in range(max_group_size)],
                    src[i : i + max_group_size],
                )

        return matrix_operator(
            mat=lambda dst, src: _grouped(dst, src, self),
            adj_mat=lambda dst, src: _grouped(dst, src, self.adj()),
            inv_mat=lambda dst, src: _grouped(dst, src, self.inv()),
            adj_inv_mat=lambda dst, src: _grouped(dst, src, self.adj().inv()),
            vector_space=self.vector_space,
            accept_guess=self.accept_guess,
            accept_list=make_list(self.accept_list),
        )

    def packed(self):
        accept_guess = self.accept_guess
        t = gpt.timer("packed")

        grid = self.vector_space[1].grid
        assert grid is not None
        n_rhs = grid.gdimensions[0]
        vector_space = tuple([x.unpacked(n_rhs) for x in self.vector_space])
        self_vector_space = self.vector_space

        def _packed(dst, src, mat, guess_id):
            t("layout")

            assert len(src) == n_rhs
            t_src = self_vector_space[1].lattice(otype=src[0].otype, cb=src[0].checkerboard())
            p_t_s = gpt.pack(t_src, fast=True)
            p_s = gpt.pack(src, fast=True)
            p_t_s.from_accelerator_buffer(p_s.to_accelerator_buffer())

            t_dst = self_vector_space[0].lattice(otype=dst[0].otype, cb=dst[0].checkerboard())
            p_t_d = gpt.pack(t_dst, fast=True)
            p_d = gpt.pack(dst, fast=True)
            if accept_guess[guess_id]:
                p_t_d.from_accelerator_buffer(p_d.to_accelerator_buffer())

            t("matrix")
            mat(t_dst, t_src)
            t("layout")
            p_d.from_accelerator_buffer(p_t_d.to_accelerator_buffer())
            t()

        return matrix_operator(
            mat=lambda dst, src: _packed(dst, src, self, 0),
            adj_mat=lambda dst, src: _packed(dst, src, self.adj(), 1),
            inv_mat=lambda dst, src: _packed(dst, src, self.inv(), 1),
            adj_inv_mat=lambda dst, src: _packed(dst, src, self.adj().inv(), 0),
            vector_space=vector_space,
            accept_guess=self.accept_guess,
            accept_list=make_list(self.accept_list),
        )

    def unary(self, u):
        if u == gpt.factor_unary.BIT_TRANS | gpt.factor_unary.BIT_CONJ:
            return self.adj()
        elif u == gpt.factor_unary.NONE:
            return self
        assert 0

    def inherit(self, parent, factory):
        if parent.inheritance is not None:
            for name, functor in parent.inheritance:
                self.__dict__[name] = functor(parent, name, factory)
            self.inheritance = parent.inheritance

        return self

    def __call__(self, first, second=None):
        assert self.mat is not None

        return_list = isinstance(first, list)
        first = gpt.util.to_list(first)

        if second is None:
            src = [gpt(x) for x in first]
        else:
            dst = first
            src = gpt.util.to_list(second)

        distribute = not self.vector_space[1].match_otype(src[0].otype)

        if second is None:
            if distribute:
                dst_vector_space = self.vector_space[0].replaced_otype(src[0].otype)
            else:
                dst_vector_space = self.vector_space[0]

            src_otype = src[0].otype
            src_grid = src[0].grid
            src_cb = src[0].checkerboard()

            n = self.lhs_length(src)

            dst = [dst_vector_space.lattice(src_grid, src_otype, src_cb) for i in range(n)]

            if self.accept_guess[0]:
                for x in dst:
                    x[:] = 0

        if self.accept_list:
            mat = self.mat
        else:

            def mat(dst, src):
                assert len(dst) == len(src)
                for idx in range(len(dst)):
                    self.mat(dst[idx], src[idx])

        if distribute:
            self.vector_space[1].otype.distribute(mat, dst, src, zero_lhs=self.accept_guess[0])
        else:
            mat(dst, src)

        if not return_list:
            return gpt.util.from_list(dst)

        return dst


class matrix_operator_product(matrix_operator):
    def __init__(self, factors):
        self.factors = factors

        first, second = factors[0], factors[-1]

        def _mat(dst, src):
            of = list(reversed(factors))
            for f in of[:-1]:
                src = f(src)
            of[-1](dst, src)

        def _adj_mat(dst, src):
            of = factors
            for f in of[:-1]:
                src = f.adj()(src)
            of[-1].adj()(dst, src)

        def _inv_mat(dst, src):
            of = factors
            for f in of[:-1]:
                src = f.inv()(src)
            of[-1].inv()(dst, src)

        def _adj_inv_mat(dst, src):
            of = list(reversed(factors))
            for f in of[:-1]:
                src = f.adj().inv()(src)
            of[-1].adj().inv()(dst, src)

        super().__init__(
            mat=_mat,
            adj_mat=_adj_mat,
            inv_mat=_inv_mat,
            adj_inv_mat=_adj_inv_mat,
            vector_space=(first.vector_space[0], second.vector_space[1]),
            accept_guess=(first.accept_guess[0], second.accept_guess[1]),
            accept_list=make_list(first.accept_list),
        )

    def __mul__(self, other):
        if isinstance(other, matrix_operator_product):
            return matrix_operator_product(self.factors + other.factors)
        elif isinstance(other, matrix_operator):
            return matrix_operator_product(self.factors + [other])
        return matrix_operator.__mul__(self, other)

    def __rmul__(self, other):
        return other.__mul__(self)
