#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2024  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.core.time import timer


class coder:
    def __init__(self, parent):
        self.parent = parent

    def __enter__(self):
        g.expr.auto_closure_stack.append(self.parent)
        return self.parent

    def __exit__(self, *args):
        g.expr.auto_closure_stack.pop()


class compiler:
    def __init__(self):
        self.compiled = False
        self.sc = []
        self.representative = None
        self.stencil = None
        self.lattices = []
        self.lattice_index = {}
        self.n_lattices = None
        self.tt = timer("compiler")
        self.verbose = g.default.is_verbose("compiler_performance")
        self.lattice_cache = []
        self.lattice_cache_index = 0

    def code(self):
        return coder(self)

    def execute(self):
        if not self.compiled:
            assert self.representative is not None
            self.tt("stencil create")
            self.stencil = g.local_stencil.matrix(
                self.representative, [tuple([0] * self.representative.grid.nd)], self.sc
            )
            self.tt()
            self.n_lattices = len(self.lattices)
            self.compiled = True

        else:
            assert len(self.lattices) == self.n_lattices

        self.tt("stencil exec")
        self.stencil(*self.lattices)
        self.tt()

        self.lattices = []
        self.lattice_index = {}
        self.lattice_cache_index = 0

        if self.verbose:
            g.message(self.tt)

    def index_of(self, lattice):
        if lattice not in self.lattice_index:
            _index = len(self.lattices)
            self.lattices.append(lattice)
            self.lattice_index[lattice] = _index

        return self.lattice_index[lattice]

    def __call__(self, first, second=None, ac=False):
        self.tt("code eval")
        if second is None:
            second = first
            if not self.compiled:
                grid, otype, return_list, nlat = first.container()
                assert nlat == 1 and not return_list
                first = g.lattice(grid, otype)
                self.lattice_cache.append(first)
            else:
                first = self.lattice_cache[self.lattice_cache_index]
                self.lattice_cache_index += 1

        if self.representative is None:
            self.representative = first
        else:
            assert (
                self.representative.grid.obj == first.grid.obj
                and self.representative.otype.__name__ == first.otype.__name__
            )

        assert second.unary == 0  # for now no traces

        _target = self.index_of(first)
        _acc = -1 if not ac else _target
        for coeff, factors in second.val:
            scf = []

            for flag, lat in factors:
                assert isinstance(lat, g.lattice)
                assert flag == 0  # later also allow for g.adj
                scf.append((self.index_of(lat), 0, 0))

            if not self.compiled:
                self.sc.append((_target, _acc, coeff, scf))

            _acc = _target

        self.tt()

        return first
