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


def assert_correct_length(a, c):
    if type(a) == list:
        for elem in a:
            assert len(elem) == c


def assert_correct_solver(x):
    if type(x) == list:
        [assert_correct_solver(elem) for elem in x]
    else:
        assert callable(x) or x is None


class mg:
    def __init__(self, mat_f, params):
        # save parameters
        self.params = params
        self.grid = params["grid"]
        self.nlevel = len(params["grid"])
        self.ncoarselevel = self.nlevel - 1
        self.finest = 0
        self.coarsest = self.nlevel - 1
        self.northo = g.util.to_list(params["northo"], self.nlevel - 1)
        self.nbasis = g.util.to_list(params["nbasis"], self.nlevel - 1)
        self.hermitian = g.util.to_list(params["hermitian"], self.nlevel - 1)
        self.savelinks = g.util.to_list(params["savelinks"], self.nlevel - 1)
        self.uselut = g.util.to_list(params["uselut"], self.nlevel - 1)
        self.preortho = g.util.to_list(params["preortho"], self.nlevel - 1)
        self.postortho = g.util.to_list(params["postortho"], self.nlevel - 1)
        self.vecstype = g.util.to_list(params["vecstype"], self.nlevel - 1)
        self.smoothsolve = g.util.to_list(params["smoothsolve"], self.nlevel - 1)
        self.setupsolve = g.util.to_list(params["setupsolve"], self.nlevel - 1)
        self.wrappersolve = g.util.to_list(params["wrappersolve"], self.nlevel - 1)
        self.distribution = g.util.to_list(params["distribution"], self.nlevel - 1)
        self.coarsestsolve = params["coarsestsolve"]

        # verbosity
        self.verbose = g.default.is_verbose("mg")

        # print prefix
        self.print_prefix = ["mg: level %d:" % i for i in range(self.nlevel)]

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
        assert_correct_length(
            [
                self.northo,
                self.nbasis,
                self.hermitian,
                self.savelinks,
                self.uselut,
                self.preortho,
                self.postortho,
                self.vecstype,
                self.smoothsolve,
                self.setupsolve,
                self.wrappersolve,
                self.distribution,
                self.nb,
            ],
            self.nlevel - 1,
        )
        assert_correct_solver([self.smoothsolve, self.setupsolve, self.coarsestsolve])
        assert type(self.coarsestsolve) != list

        # timing
        self.t_setup = [
            g.timer("mg_setup_lvl_%d" % (lvl)) for lvl in range(self.nlevel)
        ]
        self.t_solve = [
            g.timer("mg_solve_lvl_%d" % (lvl)) for lvl in range(self.nlevel)
        ]

        # temporary vectors for solve
        self.r, self.e = [None] * self.nlevel, [None] * self.nlevel
        for lvl in range(self.finest + 1, self.nlevel):
            nf_lvl = self.nf_lvl[lvl]
            self.r[lvl] = g.vcomplex(self.grid[lvl], self.nbasis[nf_lvl])
            self.e[lvl] = g.vcomplex(self.grid[lvl], self.nbasis[nf_lvl])
        self.r[self.finest] = g.vspincolor(self.grid[self.finest])

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

        # setup coarse link fields on all levels but finest
        self.A = [None] * self.nlevel
        for lvl in range(self.finest + 1, self.nlevel):
            self.A[lvl] = [
                g.mcomplex(self.grid[lvl], self.nbasis[self.nf_lvl[lvl]])
                for __ in range(9)
            ]

        # setup a history for all solvers
        self.history = [None] * self.nlevel
        for lvl in range(self.finest, self.coarsest):
            self.history[lvl] = {"smooth": [], "setup": [], "wrapper": []}
        self.history[self.coarsest] = {"coarsest": []}

        # rest of setup (call that externally?)
        self.resetup()

    def _get_slv_history(self, slv):
        return len(slv.inverter.history), slv.inverter.history[-1]

    def resetup(self, which_lvls=None):
        if which_lvls is not None:
            assert type(which_lvls) == list
            for elem in which_lvls:
                assert elem >= self.finest and elem <= self.coarsest
        else:
            which_lvls = self.lvl

        for lvl in which_lvls:
            # aliases
            t = self.t_setup[lvl]
            pp = self.print_prefix[lvl]

            # start clocks
            t("misc")

            # neighbors
            nc_lvl = self.nc_lvl[lvl]
            nf_lvl = self.nf_lvl[lvl]

            # create coarse links + operator (all but finest)
            if lvl != self.finest:
                t("create_operator")
                g.coarse.create_links(
                    self.A[lvl],
                    self.mat[nf_lvl],
                    self.basis[nf_lvl],
                    {
                        "hermitian": self.hermitian[nf_lvl],
                        "savelinks": self.savelinks[nf_lvl],
                        "uselut": self.uselut[nf_lvl],
                    },
                )
                self.mat[lvl] = g.qcd.fermion.coarse(self.A[lvl], {"level": lvl},)

                if self.verbose:
                    g.message("%s done with operator setup" % pp)

            if lvl != self.coarsest:
                t("misc")

                # aliases
                basis = self.basis[lvl]
                nb = self.nb[lvl]
                vecstype = self.vecstype[lvl]

                # pre-orthonormalize basis vectors
                if self.preortho[lvl]:
                    t("preortho")
                    g.default.push_verbose("orthogonalize", False)
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
                    if vecstype == "test":
                        psi[:] = 0.0
                        src @= v
                    elif vecstype == "null":
                        src[:] = 0.0
                        psi @= v
                    else:
                        assert 0
                    g.default.push_verbose(
                        self.setupsolve[lvl].inverter.__class__.__name__, False
                    )
                    self.setupsolve[lvl](self.mat[lvl])(psi, src)
                    g.default.pop_verbose()
                    self.history[lvl]["setup"].append(
                        self._get_slv_history(self.setupsolve[lvl])
                    )
                    v @= psi

                if self.verbose:
                    g.message("%s done finding null-space vectors" % pp)

                # post-orthonormalize basis vectors
                if self.postortho[lvl]:
                    t("postortho")
                    g.default.push_verbose("orthogonalize", False)
                    for i, v in enumerate(basis[0:nb]):
                        v /= g.norm2(v) ** 0.5
                        g.orthogonalize(v, basis[:i])
                    g.default.pop_verbose()

                    if self.verbose:
                        g.message("%s done post-orthonormalizing basis vectors" % pp)

                # chiral doubling
                t("chiral_split")
                g.split_chiral(basis)

                if self.verbose:
                    g.message("%s done doing chiral doubling" % pp)

                # block orthogonalization
                t("block_ortho")
                for i in range(self.northo[lvl]):
                    if self.verbose:
                        g.message("%s block ortho step %d" % (pp, i))
                    g.block.orthonormalize(self.grid[nc_lvl], basis)

                if self.verbose:
                    g.message("%s done block-orthonormalizing" % pp)

            t()

            if self.verbose:
                g.message("%s done with entire setup" % pp)

    def __call__(self, matrix=None):
        # ignore matrix
        mat = self.mat[self.finest]
        otype, grid, cb = None, None, None
        if type(mat) == g.matrix_operator:
            otype, grid, cb = (
                mat.otype,
                mat.grid,
                mat.cb,
            )
            mat = mat.mat

        def invert(psi, src):
            inv_lvl(psi, src, self.finest)

        def inv_lvl(psi, src, lvl):
            # aliases
            t = self.t_solve[lvl]
            pp = self.print_prefix[lvl]

            # start clocks
            t("misc")

            # assertions
            assert psi != src

            inputnorm = g.norm2(src)

            if self.verbose:
                g.message(
                    "%s starting inversion routine: psi = %g, src = %g"
                    % (pp, g.norm2(psi), g.norm2(src))
                )

            # abbreviations
            f2c = g.block.project
            c2f = g.block.promote

            # neighbors
            nc_lvl = self.nc_lvl[lvl]
            nf_lvl = self.nf_lvl[lvl]

            if lvl == self.coarsest:
                t("invert")
                g.default.push_verbose(
                    self.coarsestsolve.inverter.__class__.__name__, False
                )
                self.coarsestsolve(self.mat[lvl])(psi, src)
                g.default.pop_verbose()
                self.history[lvl]["coarsest"].append(
                    self._get_slv_history(self.coarsestsolve)
                )
            else:
                # aliases
                mat = self.mat[lvl]
                basis = self.basis[lvl]
                r = self.r[lvl]
                t = self.t_solve[lvl]

                t("copy")
                r @= src

                # fine to coarse
                t("to_coarser")
                f2c(self.r[nc_lvl], r, basis)

                if self.verbose:
                    t("output")
                    g.message(
                        "%s norm after f2c: r_c = %g" % (pp, g.norm2(self.r[nc_lvl]))
                    )

                    g.message("%s done projecting to level %d" % (pp, nc_lvl))

                # call method on next level
                t("on_coarser")
                self.e[nc_lvl][:] = 0.0
                if self.wrappersolve[lvl] is not None:
                    g.default.push_verbose(
                        self.wrappersolve[lvl].inverter.__class__.__name__, False
                    )
                    self.wrappersolve[lvl].prec = lambda dst, src: inv_lvl(
                        dst, src, nc_lvl
                    )
                    self.wrappersolve[lvl](self.mat[nc_lvl])(
                        self.e[nc_lvl], self.r[nc_lvl]
                    )
                    g.default.pop_verbose()
                    self.history[lvl]["wrapper"].append(
                        self._get_slv_history(self.wrappersolve[lvl])
                    )
                else:
                    inv_lvl(self.e[nc_lvl], self.r[nc_lvl], nc_lvl)

                if self.verbose:
                    t("output")
                    g.message("%s done calling level %d" % (pp, nc_lvl))
                    g.message(
                        "%s norms before c2f: psi = %g, e_c = %g"
                        % (pp, g.norm2(psi), g.norm2(self.e[nc_lvl]))
                    )

                # coarse to fine
                t("from_coarser")
                c2f(self.e[nc_lvl], psi, basis)

                t("residual")
                tmp = g.lattice(src)
                mat(tmp, psi)
                tmp @= src - tmp
                res_cgc = (g.norm2(tmp) / inputnorm) ** 0.5

                if self.verbose:
                    t("output")
                    g.message("%s done projecting from level %d" % (pp, nc_lvl))
                    g.message("%s norms after c2f: psi = %g" % (pp, g.norm2(psi)))

                # smooth
                t("smooth")
                g.default.push_verbose(
                    self.smoothsolve[lvl].inverter.__class__.__name__, False
                )
                self.smoothsolve[lvl](mat)(psi, src)
                g.default.pop_verbose()
                self.history[lvl]["smooth"].append(
                    self._get_slv_history(self.smoothsolve[lvl])
                )

                t("residual")
                mat(tmp, psi)
                tmp @= src - tmp
                res_smooth = (g.norm2(tmp) / inputnorm) ** 0.5

                if self.verbose:
                    g.message("%s done smoothing" % (pp))
                    g.message(
                        "%s input norm = %g, coarse residual = %g, smooth residual = %g"
                        % (pp, inputnorm, res_cgc, res_smooth)
                    )

            t()

            if self.verbose:
                g.message(
                    "%s ending inversion routine: psi = %g, src = %g"
                    % (pp, g.norm2(psi), g.norm2(src))
                )

        return g.matrix_operator(
            mat=invert, otype=otype, zero=(False, False), grid=grid, cb=cb,
        )
