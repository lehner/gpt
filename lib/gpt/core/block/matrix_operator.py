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
import gpt, cgpt
import gpt.core.operator.matrix_operator as matrix_operator
import gpt.core.block.compiler as compiler
import numpy as np


class compiled(matrix_operator):
    def __init__(self, points, implementation=None, packed_right_hand_sides=None):
        self.points = points
        self.implementation = implementation
        for p in points:
            point_grid = points[p].grid
            break

        self.point_grid = point_grid
        self.packed_right_hand_sides = packed_right_hand_sides

        if self.point_grid.cb.n == 1:
            self.M = compiler.create_stencil_operator(
                points, 0, gpt.none, self.implementation, packed_right_hand_sides
            )
        else:
            self.M = compiler.create_stencil_operator(
                points, 0, gpt.even, self.implementation, packed_right_hand_sides
            )

        super().__init__(
            mat=self.M.mat,
            vector_space=self.M.vector_space,
            accept_list=True,
        )

    def even_odd_sites_decomposed(self, parity):
        assert self.point_grid.cb.n == 1

        Mee = compiler.create_stencil_operator(
            compiler.project_points_parity(self.points, 0, gpt.even),
            0,
            gpt.even,
            self.implementation,
            self.packed_right_hand_sides,
        )
        Moo = compiler.create_stencil_operator(
            compiler.project_points_parity(self.points, 0, gpt.odd),
            1,
            gpt.odd,
            self.implementation,
            self.packed_right_hand_sides,
        )
        Meo = compiler.create_stencil_operator(
            compiler.project_points_parity(self.points, 1, gpt.even),
            1,
            gpt.even,
            self.implementation,
            self.packed_right_hand_sides,
        )
        Moe = compiler.create_stencil_operator(
            compiler.project_points_parity(self.points, 1, gpt.odd),
            0,
            gpt.odd,
            self.implementation,
            self.packed_right_hand_sides,
        )
        point_grid_eo = self.point_grid.checkerboarded(gpt.redblack)

        class even_odd_sites:
            def __init__(me):
                me.D_domain = gpt.domain.even_odd_sites(point_grid_eo, parity)
                me.C_domain = gpt.domain.even_odd_sites(point_grid_eo, parity.inv())
                if parity is gpt.even:
                    me.DD = Mee
                    me.CC = Moo
                    me.CD = Moe
                    me.DC = Meo
                else:
                    assert parity is gpt.odd
                    me.DD = Moo
                    me.CC = Mee
                    me.CD = Meo
                    me.DC = Moe
                me.DD.vector_space[1].cb = parity
                me.DD.vector_space[0].cb = parity
                me.CC.vector_space[1].cb = parity.inv()
                me.CC.vector_space[0].cb = parity.inv()
                me.CD.vector_space[1].cb = parity
                me.CD.vector_space[0].cb = parity.inv()
                me.DC.vector_space[1].cb = parity.inv()
                me.DC.vector_space[0].cb = parity

        return even_odd_sites()


class projected(matrix_operator):
    def __init__(self, map, fine_operator):
        verbose = gpt.default.is_verbose("block_operator")

        def mat(dst_coarse, src_coarse):
            src_fine = [gpt.lattice(map.basis[0]) for x in src_coarse]
            dst_fine = [gpt.lattice(map.basis[0]) for x in src_coarse]

            t0 = gpt.time()
            map.promote(src_fine, src_coarse)
            t1 = gpt.time()
            fine_operator(dst_fine, src_fine)
            t2 = gpt.time()
            map.project(dst_coarse, dst_fine)
            t3 = gpt.time()
            if verbose:
                gpt.message(
                    "coarse_operator acting on %d vector(s) in %g s (promote %g s, fine_operator %g s, project %g s)"
                    % (len(src_coarse), t3 - t0, t1 - t0, t2 - t1, t3 - t2)
                )

        otype = gpt.ot_vector_complex_additive_group(len(map.basis))
        self.map = map

        super().__init__(
            mat=mat,
            vector_space=gpt.vector_space.explicit_grid_otype(map.coarse_grid, otype),
            accept_list=True,
        )

    def compile(
        self,
        lpoints=None,
        max_point_norm=None,
        max_point_per_dimension=None,
        tolerance=None,
        nblock=None,
        packed_right_hand_sides=None,
    ):

        # accept also max length squared of points
        if lpoints is None:
            assert max_point_norm is not None and max_point_per_dimension is not None
            n, dm = max_point_norm, max_point_per_dimension
            nd = self.vector_space[0].grid.nd
            lpoints = np.mgrid[tuple([slice(-x, x + 1, 1) for x in dm])].reshape(nd, -1).T
            lpoints = lpoints[np.sum(lpoints * lpoints, axis=1) <= n]
            lpoints = [tuple([int(y) for y in x]) for x in lpoints.tolist()]
            # gpt.message(len(lpoints), "points", lpoints)

        assert lpoints is not None

        # are all points indistinguishable?
        L = np.array(self.map.coarse_grid.fdimensions)
        mod_lpoints = [str(np.mod(p + L, L)) for p in lpoints]
        min_lpoints = []
        for i in range(len(lpoints)):
            if mod_lpoints[i] not in mod_lpoints[:i]:
                min_lpoints.append(lpoints[i])

        # create irreducible points
        points = {p: gpt.mcomplex(self.map.coarse_grid, len(self.map.basis)) for p in min_lpoints}
        compiler.create(self, points, nblock=nblock)

        # get default implementation
        op = compiled(points, packed_right_hand_sides=packed_right_hand_sides)

        # test implementation
        if tolerance is not None:
            test = self.vector_space[1].lattice()
            gpt.default.push_verbose("random", False)
            gpt.random("test").cnormal(test)
            gpt.default.pop_verbose()

            # get reference implementation
            rop = compiled(
                points, implementation="reference", packed_right_hand_sides=packed_right_hand_sides
            )

            # test against reference matrix to make sure implementation is correct
            eps2_ref = tolerance
            eps2 = gpt.norm2(rop * test - op * test) / gpt.norm2(test)
            assert eps2 < eps2_ref

            # test against original matrix to make sure that points are correct
            eps2 = gpt.norm2(self * test - op * test) / gpt.norm2(test)
            if eps2 > eps2_ref:
                raise Exception(
                    f"Points {lpoints} do not seem sufficient for coarse_matrix_operator.compile: {eps2} > {eps2_ref}"
                )

        return op
