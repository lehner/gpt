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


##
# domain:
# - explicit set of spacetime points that can be embedded in a full lattice
#
# vector_space:
# - set of spacetime points and object type
# - implemented through tuple (grid, otype, checkerboard)
# - all components can be left undefined and inferred from
#   a source vector
class general:
    def __init__(self, grid, otype, cb):
        self.grid = grid
        self.otype = otype
        self.cb = cb

    def converted(self, precision):
        return general(self.grid.converted(precision), self.otype, self.cb)

    def lattice(self, grid=None, otype=None, cb=None):
        if self.grid is not None:
            grid = self.grid
        if self.otype is not None:
            otype = self.otype
        if self.cb is not None:
            cb = self.cb

        l = gpt.lattice(grid, otype)
        if cb is not None and grid.cb.n != 1:
            l.checkerboard(cb)
        return l

    def match_otype(self, otype):
        return otype is None or self.otype is None or otype.__name__ == self.otype.__name__

    def replaced_otype(self, otype):
        return general(self.grid, otype, self.cb)

    def clone(self):
        return general(self.grid, self.otype, self.cb)


##
# short-hand
def implicit():
    return general(None, None, None)


def explicit_grid(grid):
    return general(grid, None, None)


def explicit_domain_otype(domain, otype):
    return general(domain.grid, otype, None)


def explicit_grid_otype_checkerboard(grid, otype, cb):
    return general(grid, otype, cb)


def explicit_grid_otype(grid, otype):
    return general(grid, otype, None)


def explicit_lattice(lat):
    return general(lat.grid, lat.otype, lat.checkerboard())
