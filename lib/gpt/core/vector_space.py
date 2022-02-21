#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


def get_lattice(similar_to):
    if similar_to is None:
        return None
    elif isinstance(similar_to, gpt.lattice):
        return similar_to
    else:
        return similar_to[1]


class base_explicit_otype:
    def get_otype(self, similar_to):
        if type(similar_to) == tuple and not similar_to[0]:
            return similar_to[1].otype
        else:
            return self.otype

    def evaluate(self, matrix, lhs, rhs, zero_lhs):
        if self.otype.__name__ == rhs[0].otype.__name__:
            matrix(lhs, rhs)
        else:
            # zero_lhs only if you create temporary new lhs variables
            self.otype.distribute(matrix, lhs, rhs, zero_lhs=zero_lhs)

    def type_match(self, rhs):
        if self.otype.__name__ == rhs.otype.__name__:
            return True, rhs
        else:
            return False, rhs


class base_implicit_otype:
    def evaluate(self, matrix, lhs, rhs, zero_lhs):
        matrix(lhs, rhs)

    def type_match(self, rhs):
        return True, rhs


class explicit_domain_otype(base_explicit_otype):
    def __init__(self, domain, otype):
        self.domain = domain
        self.otype = otype

    @property
    def grid(self):
        return self.domain.grid

    def converted(self, precision):
        return explicit_domain_otype(self.domain.converted(precision), self.otype)

    def lattice(self, similar_to):
        return self.domain.lattice(self.get_otype(similar_to))


class explicit_grid_otype_checkerboard(base_explicit_otype):
    def __init__(self, grid, otype, cb):
        self.grid = grid
        self.otype = otype
        self.cb = cb

    def converted(self, precision):
        return explicit_grid_otype_checkerboard(
            self.grid.converted(precision), self.otype, self.cb
        )

    def lattice(self, similar_to):
        l = gpt.lattice(self.grid, self.get_otype(similar_to))
        l.checkerboard(self.cb)
        return l


class explicit_grid_otype(base_explicit_otype):
    def __init__(self, grid, otype):
        self.grid = grid
        self.otype = otype

    def converted(self, precision):
        return explicit_grid_otype(self.grid.converted(precision), self.otype)

    def lattice(self, similar_to):
        l = gpt.lattice(self.grid, self.get_otype(similar_to))
        ls = get_lattice(similar_to)
        if ls is not None:
            l.checkerboard(ls.checkerboard())
        return l


class explicit_grid(base_implicit_otype):
    def __init__(self, grid):
        self.grid = grid

    def converted(self, precision):
        return explicit_grid(self.grid.converted(precision))

    def lattice(self, similar_to):
        ls = get_lattice(similar_to)
        l = gpt.lattice(self.grid, ls.otype)
        l.checkerboard(ls.checkerboard())
        return l


class implicit(base_implicit_otype):
    def converted(self, precision):
        return self

    def lattice(self, similar_to):
        ls = get_lattice(similar_to)
        l = gpt.lattice(ls)
        l.checkerboard(ls.checkerboard())
        return l


# short-hand
def explicit_lattice(lat):
    return explicit_grid_otype_checkerboard(lat.grid, lat.otype, lat.checkerboard())
