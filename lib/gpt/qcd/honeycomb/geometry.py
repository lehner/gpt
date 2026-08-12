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
from itertools import product


class geometry:
    def get_coor(self, x):
        if abs(int(x[0]) - x[0]) < 1e-13:
            sub_grid = 0
        else:
            sub_grid = 1

        # U[sub_grid][idx][x - sub_offset[subset[x]]]
        x = np.array(x) - self.sub_offset[sub_grid]
        assert all(abs(int(y) - y) < 1e-13 for y in x)
        x = tuple(int(y) for y in x)
        return sub_grid, x

    def get_link(self, x_src, x_dst):
        # get direction
        dx = np.array(x_dst) - np.array(x_src)
        dx = tuple(float(x) for x in dx)

        # is link
        if dx in self.positive:
            dag = False
            idx = self.positive.index(dx)
            x = x_src
        else:
            dag = True
            dx = tuple(float(-x) for x in dx)
            idx = self.positive.index(dx)
            x = x_dst

        sub_grid, shift = self.get_coor(x)
        return sub_grid, idx, dag, shift

    def __init__(self):
        positive = [
            # original 4
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
            # all positive
            (0.5, 0.5, 0.5, 0.5),
            # one negative
            (-0.5, 0.5, 0.5, 0.5),
            (0.5, -0.5, 0.5, 0.5),
            (0.5, 0.5, -0.5, 0.5),
            (0.5, 0.5, 0.5, -0.5),
            # two negative (symmetries allow first to always be positive)
            (0.5, 0.5, -0.5, -0.5),
            (0.5, -0.5, 0.5, -0.5),
            (0.5, -0.5, -0.5, 0.5),
        ]

        assert len(positive) == 12
        self.positive = positive

        # get hypercubic link indices
        self.link_hypercube_indices = [0, 1, 2, 3]  # [5,6,7,8], [4,9,10,11]

        # is the direction changing sub-grids?
        cross_grids = [0] * 4 + [1] * 8
        self.cross_grids = cross_grids

        # how many link variables do we need?
        self.number_of_directions = len(positive)
        self.number_of_subsets = 2
        self.number_of_link_fields = self.number_of_subsets * self.number_of_directions

        # offsets between sub-grids
        sub_offset = [np.array([0] * 4), np.array([0.5] * 4)]
        self.sub_offset = sub_offset

        # point at x = sub_offset corresponds to origin within sub-grid
        # now need to map every point pair of x on the grid to x + positive[i] to link
        # U[subset[x]][i][x - sub_offset[subset[x]]]

        # directions
        directions = positive + [tuple(-d for d in x) for x in positive]
        self.directions = directions

        # next pre-compute plaquette instructions
        # all plaquettes must have one on-axis and two off-axis components ; mu picks the axis
        plaquettes = []
        for mu in range(4):
            free = [nu for nu in range(4) if nu != mu]
            for signs in product([+1, -1], repeat=3):
                # axis vector
                v_axis = tuple(1 if nu == mu else 0 for nu in range(4))
                # two diagonal links (scaled by 2)
                d1 = tuple(-0.5 if nu == mu else 0.5 * signs[free.index(nu)] for nu in range(4))
                d2 = tuple(-v_axis[i] - d1[i] for i in range(4))
                # add
                plaquettes.append((v_axis, d1, d2))

        # check lattice
        assert len(plaquettes) == 32

        # now create instructions
        plaquette_instructions = []
        for i, j, k in plaquettes:
            m = [i, j, k]

            triangle = []
            pos = np.array([0] * 4, dtype=np.float64)
            for l in range(3):
                pos += np.array(m[l], dtype=np.float64)
                triangle.append(tuple(float(x) for x in pos))

            instructions = []
            for l in range(3):
                sub, idx, dag, shift = self.get_link(triangle[l], triangle[(l + 1) % 3])
                instructions.append((sub, dag, shift, idx))
            plaquette_instructions.append(instructions)

            instructions = []
            for l in range(3):
                sub, idx, dag, shift = self.get_link(
                    triangle[l] + sub_offset[1], triangle[(l + 1) % 3] + sub_offset[1]
                )
                instructions.append((sub, dag, shift, idx))
            plaquette_instructions.append(instructions)

        self.plaquette_instructions = plaquette_instructions
