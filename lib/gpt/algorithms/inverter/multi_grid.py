#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann (daniel.richtmann@ur.de)
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
import sys
import copy


def get_slv_name(slv):
    if hasattr(slv, "inverter"):
        return slv.inverter.__class__.__name__
    else:
        return slv.__class__.__name__


def get_slv_history(slv):
    s = slv.inverter if hasattr(slv, "inverter") else slv
    if s.history is not None:
        return len(s.history), s.history[-1]
    else:
        return 0, 0.0


class multi_grid_setup:
    def __init__(self, mat_f, params):
        # save parameters
        self.params = params

        # fine grid from fine matrix
        if issubclass(type(mat_f), g.matrix_operator):
            self.grid = [mat_f.grid[1]]
        else:
            self.grid = [mat_f.grid]

        # grid sizes - allow specifying in two ways
        if "grid" in params:
            self.grid.extend(params["grid"])
        elif "block_size" in params:
            for i, bs in enumerate(params["block_size"]):
                assert type(bs) == list
                self.grid.append(g.block.grid(self.grid[i], bs))
        else:
            assert 0

        # dependent sizes
        self.nlevel = len(self.grid)
        self.ncoarselevel = self.nlevel - 1
        self.finest = 0
        self.coarsest = self.nlevel - 1

        # other parameters
        self.nblockortho = g.util.to_list(params["n_block_ortho"], self.nlevel - 1)
        self.check_blockortho = g.util.to_list(
            params["check_block_ortho"], self.nlevel - 1
        )
        self.nbasis = g.util.to_list(params["n_basis"], self.nlevel - 1)
        self.make_hermitian = g.util.to_list(params["make_hermitian"], self.nlevel - 1)
        self.save_links = g.util.to_list(params["save_links"], self.nlevel - 1)
        self.npreortho = g.util.to_list(params["n_pre_ortho"], self.nlevel - 1)
        self.npostortho = g.util.to_list(params["n_post_ortho"], self.nlevel - 1)
        self.vector_type = g.util.to_list(params["vector_type"], self.nlevel - 1)
        self.distribution = g.util.to_list(params["distribution"], self.nlevel - 1)
        self.solver = g.util.to_list(params["solver"], self.nlevel - 1)

        # verbosity
        self.verbose = g.default.is_verbose("multi_grid_setup")

        # print prefix
        self.print_prefix = ["mg_setup: level %d:" % i for i in range(self.nlevel)]

        # easy access to current level and neighbors
        self.lvl = [i for i in range(self.nlevel)]
        self.nf_lvl = [i - 1 for i in range(self.nlevel)]
        self.nc_lvl = [i + 1 for i in range(self.nlevel)]
        self.nf_lvl[self.finest] = None
        self.nc_lvl[self.coarsest] = None

        # halved nbasis
        self.nb = []
        for lvl, b in enumerate(self.nbasis):
            assert b % 2 == 0
            self.nb.append(b // 2)

        # assertions
        assert self.nlevel >= 2
        assert g.util.entries_have_length(
            [
                self.nblockortho,
                self.nbasis,
                self.make_hermitian,
                self.save_links,
                self.npreortho,
                self.npostortho,
                self.vector_type,
                self.distribution,
                self.solver,
                self.nb,
            ],
            self.nlevel - 1,
        )

        # timing
        self.t = [g.timer("mg_setup_lvl_%d" % (lvl)) for lvl in range(self.nlevel)]

        # matrices (coarse ones initialized later)
        self.mat = [mat_f] + [None] * (self.nlevel - 1)

        # setup random basis vectors on all levels but coarsest
        self.basis = [None] * self.nlevel
        for lvl, grid in enumerate(self.grid):
            if lvl == self.coarsest:
                continue
            elif lvl == self.finest:
                self.basis[lvl] = [g.vspincolor(grid) for __ in range(self.nbasis[lvl])]
            else:
                self.basis[lvl] = [
                    g.vcomplex(grid, self.nbasis[self.nf_lvl[lvl]])
                    for __ in range(self.nbasis[lvl])
                ]
            self.distribution[lvl](self.basis[lvl][0 : self.nb[lvl]])

        # open bc for basis vectors if finest level matrix is open bc
        # the way I retrieve this info is rather ugly -> TODO?
        if self.mat[self.finest].params["boundary_phases"][-1] == 0.0:
            for lvl, grid in enumerate(self.grid):
                if lvl != self.coarsest:
                    # TODO: the following needs to be part of a more abstract
                    # interface, should not refer to a qcd module function explicitly
                    # here!  Do this in next iteration of MG interface (inv.sequence, ...).
                    g.qcd.fermion.apply_open_boundaries(
                        self.basis[lvl][0 : self.nb[lvl]]
                    )

        # setup a block map on all levels but coarsest
        self.blockmap = [None] * self.nlevel
        for lvl in self.lvl:
            if lvl == self.coarsest:
                continue
            else:
                self.blockmap[lvl] = g.block.map(
                    self.grid[self.nc_lvl[lvl]], self.basis[lvl]
                )

        # setup coarse link fields on all levels but finest
        self.A = [None] * self.nlevel
        for lvl in range(self.finest + 1, self.nlevel):
            self.A[lvl] = [
                g.mcomplex(self.grid[lvl], self.nbasis[self.nf_lvl[lvl]])
                for __ in range(9)
            ]

        # setup a solver history
        self.history = [[None]] * (self.nlevel - 1)

        # rest of setup
        self.__call__()

    def __getitem__(self, level):
        return (self.grid[level + 1], self.basis[level])

    # TODO This needs to be able to get a parameter dict and act accordingly
    def __call__(self, which_lvls=None):
        if which_lvls is not None:
            assert type(which_lvls) == list
            for elem in which_lvls:
                assert elem >= self.finest and elem <= self.coarsest
        else:
            which_lvls = self.lvl

        for lvl in which_lvls:
            if lvl == self.coarsest:
                continue

            # aliases
            t = self.t[lvl]
            pp = self.print_prefix[lvl]

            # start clocks
            t("misc")

            # neighbors
            nc_lvl = self.nc_lvl[lvl]

            # aliases
            basis = self.basis[lvl]
            blockmap = self.blockmap[lvl]
            nb = self.nb[lvl]
            vector_type = self.vector_type[lvl]

            # pre-orthonormalize basis vectors globally
            t("pre_ortho")
            g.default.push_verbose("orthogonalize", False)
            for n in range(self.npreortho[lvl]):
                if self.verbose:
                    g.message("%s pre ortho step %d" % (pp, n))
                for i, v in enumerate(basis[0:nb]):
                    v /= g.norm2(v) ** 0.5
                    g.orthogonalize(v, basis[:i])
            g.default.pop_verbose()

            if self.verbose:
                g.message("%s done pre-orthonormalizing basis vectors" % pp)

            # find near-null vectors
            t("find_null_vecs")
            src, psi = g.copy(basis[0]), g.copy(basis[0])
            for i, v in enumerate(basis[0:nb]):
                if vector_type == "test":
                    psi[:] = 0.0
                    src @= v
                elif vector_type == "null":
                    src[:] = 0.0
                    psi @= v
                else:
                    assert 0
                g.default.push_verbose(get_slv_name(self.solver[lvl]), False)
                self.solver[lvl](self.mat[lvl])(psi, src)
                g.default.pop_verbose()
                self.history[lvl].append(get_slv_history(self.solver[lvl]))
                v @= psi

            if self.verbose:
                g.message("%s done finding null-space vectors" % pp)

            # post-orthonormalize basis vectors globally
            t("post_ortho")
            g.default.push_verbose("orthogonalize", False)
            for n in range(self.npostortho[lvl]):
                if self.verbose:
                    g.message("%s post ortho step %d" % (pp, n))
                for i, v in enumerate(basis[0:nb]):
                    v /= g.norm2(v) ** 0.5
                    g.orthogonalize(v, basis[:i])
            g.default.pop_verbose()

            if self.verbose:
                g.message("%s done post-orthonormalizing basis vectors" % pp)

            # chiral doubling
            t("chiral_split")
            g.coarse.split_chiral(basis)

            if self.verbose:
                g.message("%s done doing chiral doubling" % pp)

            # orthonormalize blocks
            t("block_ortho")
            for i in range(self.nblockortho[lvl]):
                if self.verbose:
                    g.message("%s block ortho step %d" % (pp, i))
                blockmap.orthonormalize()

            if self.check_blockortho:
                t("block_ortho_check")
                blockmap.check_orthogonality()

            if self.verbose:
                g.message("%s done block-orthonormalizing" % pp)

            # create coarse links + operator
            t("create_operator")
            g.coarse.create_links(
                self.A[nc_lvl],
                self.mat[lvl],
                self.basis[lvl],
                {
                    "make_hermitian": self.make_hermitian[lvl],
                    "save_links": self.save_links[lvl],
                },
            )
            self.mat[nc_lvl] = g.qcd.fermion.coarse(self.A[nc_lvl], {"level": nc_lvl})

            if self.verbose:
                g.message("%s done setting up next coarser operator" % pp)

            t()

            if self.verbose:
                g.message("%s done with entire setup" % pp)


class multi_grid:
    @g.params_convention(make_hermitian=False, save_links=True)
    def __init__(self, coarse_inverter, coarse_grid, basis, params):
        self.params = params
        self.coarse_inverter = coarse_inverter
        self.coarse_grid = coarse_grid
        self.verbose = g.default.is_verbose("multi_grid_inverter")
        self.basis = basis

    def __call__(self, mat):

        assert isinstance(mat, g.matrix_operator)
        otype, fine_grid, cb = mat.otype, mat.grid, mat.cb

        bm = g.block.map(self.coarse_grid, self.basis)

        A = [g.mcomplex(self.coarse_grid, len(self.basis)) for i in range(9)]

        g.coarse.create_links(
            A,
            mat,
            self.basis,
            {
                "make_hermitian": self.params["make_hermitian"],
                "save_links": self.params["save_links"],
            },
        )

        # level == 0 <> otype != mcomplex
        level = 1 if isinstance(otype, g.ot_matrix_singlet) else 0
        cmat = g.qcd.fermion.coarse(A, {"level": level})
        cinv = self.coarse_inverter(cmat)

        def inv(dst, src):
            assert dst != src
            if self.verbose:
                g.message(f"Enter grid {self.coarse_grid}")
                t0 = g.time()

            g.eval(dst, bm.promote * cinv * bm.project * src)

            if self.verbose:
                t1 = g.time()
                g.message(
                    f"Back to grid {fine_grid[0]}, spent {t1-t0} seconds on coarse grid"
                )

        return g.matrix_operator(
            mat=inv,
            inv_mat=mat,
            adj_mat=None,
            adj_inv_mat=None,
            otype=otype,
            accept_guess=(True, False),
            accept_list=True,
            grid=fine_grid,
            cb=cb,
        )
