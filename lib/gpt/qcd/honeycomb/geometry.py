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


class geometry:
    def __init__(self):
        # construct a few points in two hypercubes
        points = [
            np.array([x, y, z, t])
            for x in range(0, 3)
            for y in range(0, 3)
            for z in range(0, 3)
            for t in range(0, 3)
        ] + [
            np.array([x + 0.5, y + 0.5, z + 0.5, t + 0.5])
            for x in range(0, 3)
            for y in range(0, 3)
            for z in range(0, 3)
            for t in range(0, 3)
        ]

        # get all directions
        directions = set([])
        for x in points:
            for y in points:
                d = x - y
                n2 = d @ d
                if n2 == 1:
                    d = tuple(float(z) for z in d)
                    directions.add(d)
        directions = list(directions)

        assert len(directions) == 24
        self.directions = directions

        # get all positive directions
        positive = set([])
        for d in directions:
            keep = True
            for mu in range(4):
                if d[mu] < 0:
                    keep = False
                elif d[mu] > 0:
                    break
            if keep:
                positive.add(d)
        positive = list(positive)

        assert len(positive) == 12
        self.positive = positive

        # now need to map each index in directions to the corresponding positive index and direction
        map_positive = []
        for i in range(len(directions)):
            d = directions[i]
            if d in positive:
                map_positive.append((True, positive.index(d)))
            else:
                dm = tuple(-x for x in d)
                map_positive.append((False, positive.index(dm)))

        self.map_positive = map_positive

        # is the positive direction changing sub-grids?
        positive_subset = [
            [subset if int(pos0[0]) == pos0[0] else 1 - subset for pos0 in positive]
            for subset in [0, 1]
        ]
        self.positive_subset = positive_subset

        # how many link variables do we need?
        self.number_of_link_fields = len(directions)
        self.number_of_subsets = 2

        # offsets between sub-grids
        sub_offset = [np.array([0] * 4), np.array([0.5] * 4)]
        self.sub_offset = sub_offset

        # now need to map every point pair of x on the grid to x + positive[i] to link
        # U[subset[x]][i][x - sub_offset[subset[x]]]

        # next pre-compute plaquette instructions
        plaquettes = []
        for i in range(len(directions)):
            A = directions[i]
            for j in range(i):
                B = directions[j]
                for k in range(j):
                    C = directions[k]
                    if sum((A[mu] + B[mu] + C[mu]) ** 2 for mu in range(4)) == 0:
                        plaquettes.append((i, j, k))

        assert len(plaquettes) == 32

        plaquette_instructions = [[], []]
        for i, j, k in plaquettes:
            m = [map_positive[i], map_positive[j], map_positive[k]]
            for o in [0, 1]:
                pos = np.array([0, 0, 0, 0], dtype=np.float64) + sub_offset[o]
                instructions = []
                for l in range(3):
                    if m[l][0]:
                        pos0 = np.copy(pos)
                        pos += np.array(positive[m[l][1]], dtype=np.float64)
                        dag = False
                        # print(f"U({pos0} -> {pos}) = U_{{{m[l][1]}}}({pos0})")
                    else:
                        pos -= np.array(positive[m[l][1]], dtype=np.float64)
                        pos0 = np.copy(pos)
                        dag = True
                        # print(f"U({pos} -> {pos0}) = U^dag_{{{m[l][1]}}}({pos0})")

                    subset = 0 if int(pos0[0]) == pos0[0] else 1
                    if subset == 1:
                        pos0 -= sub_offset[1]
                        # print("U",subset,dag,pos0,m[l][1])
                    instructions.append((subset, dag, pos0, m[l][1]))
            plaquette_instructions[o].append(instructions)

        self.plaquette_instructions = plaquette_instructions[0] + plaquette_instructions[1]
