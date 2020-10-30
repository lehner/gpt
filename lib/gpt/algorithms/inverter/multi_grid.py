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


class setup:
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


class inverter:
    def __init__(self, setup, params):
        # save input
        self.setup = setup
        self.params = params

        # aliases
        s = self.setup
        par = self.params

        # parameters
        self.smooth_solver = g.util.to_list(par["smooth_solver"], s.nlevel - 1)
        self.wrapper_solver = g.util.to_list(par["wrapper_solver"], s.nlevel - 2)
        self.coarsest_solver = par["coarsest_solver"]

        # verbosity
        self.verbose = g.default.is_verbose("multi_grid_inverter")

        # print prefix
        self.print_prefix = ["mg: level %d:" % i for i in range(s.nlevel)]

        # assertions
        assert g.util.entries_have_length([self.smooth_solver], s.nlevel - 1)
        assert g.util.entries_have_length([self.wrapper_solver], s.nlevel - 2)
        assert g.util.is_callable(
            [self.smooth_solver, self.coarsest_solver, self.wrapper_solver]
        )
        assert type(self.coarsest_solver) != list
        assert not g.util.all_have_attribute(self.wrapper_solver, "inverter")

        # timing
        self.t = [g.timer("mg_solve_lvl_%d" % (lvl)) for lvl in range(s.nlevel)]

        # temporary vectors
        self.r, self.e = [None] * s.nlevel, [None] * s.nlevel
        for lvl in range(s.finest + 1, s.nlevel):
            nf_lvl = s.nf_lvl[lvl]
            self.r[lvl] = g.vcomplex(s.grid[lvl], s.nbasis[nf_lvl])
            self.e[lvl] = g.vcomplex(s.grid[lvl], s.nbasis[nf_lvl])
        self.r[s.finest] = g.vspincolor(s.grid[s.finest])

        # setup a history for all solvers
        self.history = [None] * s.nlevel
        for lvl in range(s.finest, s.coarsest):
            self.history[lvl] = {"smooth": [], "wrapper": []}
        self.history[s.coarsest] = {"coarsest": []}

    def __call__(self, matrix=None):

        # ignore matrix

        s = self.setup
        otype = (self.r[s.finest].otype, self.r[s.finest].otype)
        grid = (self.r[s.finest].grid, self.r[s.finest].grid)
        cb = (self.r[s.finest].checkerboard(), self.r[s.finest].checkerboard())

        def inv_lvl(psi, src, lvl):
            # assertions
            assert psi != src
            assert type(src) != list

            # neighbors
            nc_lvl = s.nc_lvl[lvl]

            # aliases
            t = self.t[lvl]
            r = self.r[lvl]
            pp = self.print_prefix[lvl]
            r_c = self.r[nc_lvl] if lvl != s.coarsest else None
            e_c = self.e[nc_lvl] if lvl != s.coarsest else None
            mat_c = s.mat[nc_lvl] if lvl != s.coarsest else None
            mat = s.mat[lvl]
            bm = s.blockmap[lvl]
            slv_s = self.smooth_solver[lvl] if lvl != s.coarsest else None
            slv_w = self.wrapper_solver[lvl] if lvl <= s.coarsest - 2 else None
            slv_c = self.coarsest_solver if lvl == s.coarsest else None

            # start clocks
            t("misc")

            if self.verbose:
                g.message(
                    "%s starting inversion routine: psi = %g, src = %g"
                    % (pp, g.norm2(psi), g.norm2(src))
                )

            inputnorm = g.norm2(src)

            if lvl == s.coarsest:
                t("invert")
                g.default.push_verbose(get_slv_name(slv_c), False)
                slv_c(mat)(psi, src)
                g.default.pop_verbose()
                self.history[lvl]["coarsest"].append(get_slv_history(slv_c))
            else:
                t("copy")
                r @= src

                # fine to coarse
                t("to_coarser")
                bm.project(r_c, r)

                if self.verbose:
                    t("output")
                    g.message(
                        "%s done calling f2c: r_c = %g, r = %g"
                        % (pp, g.norm2(r_c), g.norm2(r))
                    )

                # call method on next level
                t("on_coarser")
                e_c[:] = 0.0
                if slv_w is not None and lvl < s.coarsest - 1:

                    def prec(matrix):
                        def ignore_mat(dst_p, src_p):
                            inv_lvl(dst_p, src_p, nc_lvl)

                        # return g.matrix_operator(ignore_mat)
                        return ignore_mat

                    g.default.push_verbose(get_slv_name(slv_w), False)
                    slv_w.modified(prec=prec)(mat_c)(e_c, r_c)
                    g.default.pop_verbose()
                    self.history[lvl]["wrapper"].append(get_slv_history(slv_w))
                else:
                    inv_lvl(e_c, r_c, nc_lvl)

                if self.verbose:
                    t("output")
                    g.message(
                        "%s done calling coarser level: e_c = %g, r_c = %g"
                        % (pp, g.norm2(e_c), g.norm2(r_c))
                    )

                # coarse to fine
                t("from_coarser")
                bm.promote(psi, e_c)

                if self.verbose:
                    t("output")
                    g.message(
                        "%s done calling c2f: psi = %g, e_c = %g"
                        % (pp, g.norm2(psi), g.norm2(e_c))
                    )

                t("residual")
                tmp = g.lattice(src)
                mat(tmp, psi)
                tmp @= src - tmp
                res_cgc = (g.norm2(tmp) / inputnorm) ** 0.5

                # smooth
                t("smooth")
                g.default.push_verbose(get_slv_name(slv_s), False)
                slv_s(mat)(psi, src)
                g.default.pop_verbose()
                self.history[lvl]["smooth"].append(get_slv_history(slv_s))

                t("residual")
                mat(tmp, psi)
                tmp @= src - tmp
                res_smooth = (g.norm2(tmp) / inputnorm) ** 0.5

                if self.verbose:
                    t("output")
                    g.message(
                        "%s done smoothing: input norm = %g, coarse residual = %g, smooth residual = %g"
                        % (pp, inputnorm, res_cgc, res_smooth)
                    )

            t()

            if self.verbose:
                t("output")
                g.message(
                    "%s ending inversion routine: psi = %g, src = %g"
                    % (pp, g.norm2(psi), g.norm2(src))
                )
                t()

        return g.matrix_operator(
            mat=lambda dst, src: inv_lvl(dst, src, self.setup.finest),
            adj_mat=None,
            inv_mat=self.setup.mat[self.setup.finest],
            adj_inv_mat=None,
            otype=otype,
            accept_guess=(True, False),
            grid=grid,
            cb=cb,
        )
