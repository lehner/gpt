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


def make_param_list(a, c):
    if type(a) == list:
        return a
    return [a] * c


def assert_correct_length(a, c):
    if type(a) == list:
        for elem in a:
            assert len(elem) == c


class mg:
    def __init__(self, mat_f, params):
        # save parameters
        self.params = params
        self.grid = params["grid"]
        self.nlevel = len(params["grid"])
        self.ncoarselevel = self.nlevel - 1
        self.finest = 0
        self.coarsest = self.nlevel - 1
        self.northo = make_param_list(params["northo"], self.nlevel - 1)
        self.nbasis = make_param_list(params["nbasis"], self.nlevel - 1)
        self.hermitian = make_param_list(params["hermitian"], self.nlevel - 1)
        self.vecstype = make_param_list(params["vecstype"], self.nlevel - 1)
        self.presmooth = make_param_list(params["presmooth"], self.nlevel - 1)
        self.postsmooth = make_param_list(params["postsmooth"], self.nlevel - 1)
        self.setupsolve = make_param_list(params["setupsolve"], self.nlevel - 1)
        self.coarsestsolve = params["coarsestsolve"]

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
                self.vecstype,
                self.presmooth,
                self.postsmooth,
                self.setupsolve,
                self.nb,
            ],
            self.nlevel - 1,
        )
        assert type(self.coarsestsolve) != list

        # timing
        self.t_setup = [g.timer() for __ in range(self.nlevel)]
        self.t_solve = [g.timer() for __ in range(self.nlevel)]

        # rng
        self.rng = g.random("multigrid")

        # temporary vectors for solve
        self.r, self.e = [None] * self.nlevel, [None] * self.nlevel
        for lvl in range(self.finest + 1, self.nlevel):
            nf_lvl = self.nf_lvl[lvl]
            self.r[lvl] = g.vcomplex(self.grid[lvl], self.nbasis[nf_lvl])
            self.e[lvl] = g.vcomplex(self.grid[lvl], self.nbasis[nf_lvl])
        self.r[self.finest] = g.vspincolor(self.grid[self.finest])

        # matrices (coarse ones initialized later)
        self.mat = [mat_f] + [None] * (self.nlevel - 1)

        # setup basis vectors on all levels but coarsest
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
        self.rng.cnormal(self.basis)

        # setup coarse link fields on all levels but finest
        self.A = [None] * self.nlevel
        for lvl in range(self.finest + 1, self.nlevel):
            self.A[lvl] = [
                g.mcomplex(self.grid[lvl], self.nbasis[lvl - 1]) for __ in range(9)
            ]

        # rest of setup (call that externally?)
        self.resetup()

    def resetup(self):
        for lvl in self.lvl:
            # aliases
            t = self.t_setup[lvl]

            # start clocks
            t.start("total")
            t.start("misc")

            # abbreviations
            s = g.qcd.fermion.solver
            a = g.algorithms.iterative

            # neighbors
            nc_lvl = self.nc_lvl[lvl]
            nf_lvl = self.nf_lvl[lvl]

            t.stop("misc")

            # create coarse links + operator (all but finest)
            if lvl != self.finest:
                t.start("create_operator")
                g.coarse.create_links(self.A[lvl], self.mat[nf_lvl], self.basis[nf_lvl])
                self.mat[lvl] = g.qcd.fermion.coarse(
                    self.A[lvl], {"hermitian": self.hermitian[nf_lvl], "level": lvl},
                )
                t.stop("create_operator")
                g.message("Done with operator setup on level %d" % lvl)

            if lvl != self.coarsest:
                t.start("misc")

                # aliases
                basis = self.basis[lvl]
                nb = self.nb[lvl]
                vecstype = self.vecstype[lvl]

                # setup solver
                slv_setup = s.propagator(
                    s.inv_direct(self.mat[lvl], self.setupsolve[lvl])
                )

                # TODO
                # g.unsplit_chiral(basis)

                t.stop("misc")

                # find near-null vectors
                t.start("find_null_vecs")
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
                    slv_setup(psi, src)
                    v @= psi
                t.stop("find_null_vecs")
                g.message("Done finding null-space vectors on level %d" % lvl)

                # chiral doubling
                t.start("chiral_split")
                g.split_chiral(basis)
                t.stop("chiral_split")
                g.message("Done doing chiral doubling on level %d" % lvl)

                # block orthogonalization
                t.start("block_ortho")
                for i in range(self.northo[lvl]):
                    g.message("Block ortho step %d on level %d" % (i, lvl))
                    g.block.orthonormalize(self.grid[nc_lvl], basis)
                t.stop("block_ortho")
                g.message("Done block-orthonormalizing on level %d" % lvl)

            t.stop("total")

            g.message("Done with entire setup on level %d" % lvl)

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

            # start clock
            t.start("total")

            # assertions
            assert psi != src

            g.message(
                "Starting inversion routine on level %d: psi = %g, src = %g"
                % (lvl, g.norm2(psi), g.norm2(src))
            )

            # abbreviations
            s = g.qcd.fermion.solver
            f2c = g.block.project
            c2f = g.block.promote

            # neighbors
            nc_lvl = self.nc_lvl[lvl]
            nf_lvl = self.nf_lvl[lvl]

            if lvl == self.coarsest:
                t.start("invert")
                # TODO enable eo (requires work in Grid)
                slv_coarsest = s.propagator(
                    s.inv_direct(self.mat[lvl], self.coarsestsolve)
                )
                slv_coarsest(psi, src)
                t.stop("invert")
            else:
                # aliases
                mat = self.mat[lvl]
                basis = self.basis[lvl]
                presmooth = self.presmooth[lvl]
                postsmooth = self.postsmooth[lvl]
                r = self.r[lvl]
                t = self.t_solve[lvl]

                # run optional pre-smoother
                # TODO enable eo (requires work in Grid)
                # TODO check algorithm regarding presmoothing
                if False:
                    t.start("presmooth")
                    tmp, mmtmp = g.lattice(src), g.lattice(src)
                    tmp[:] = 0
                    slv_presmooth = s.propagator(s.inv_direct(mat, presmooth))
                    slv_presmooth(tmp, src)
                    mat.M(mmtmp, tmp)
                    r @= src - mmtmp
                    t.stop("presmooth")
                else:
                    t.start("copy")
                    r @= src
                    t.stop("copy")

                g.message("Done presmoothing on level %d" % (lvl))

                # fine to coarse
                t.start("tocoarse")
                f2c(self.r[nc_lvl], r, basis)
                t.stop("tocoarse")

                g.message("Done projecting from level %d to level %d" % (lvl, nc_lvl))

                # call method on next level TODO wrap by solver for k-cycle
                t.start("nextlevel")
                inv_lvl(self.e[nc_lvl], self.r[nc_lvl], nc_lvl)
                t.stop("nextlevel")

                g.message("Done calling level %d from level %d" % (nc_lvl, lvl))

                # coarse to fine
                t.start("fromcoarse")
                c2f(self.e[nc_lvl], psi, basis)
                t.stop("fromcoarse")

                g.message("Done projecting from level %d to level %d" % (nc_lvl, lvl))

                # run optional pre-smoother TODO make optional
                t.start("postsmooth")
                # TODO enable eo (requires work in Grid)
                slv_postsmooth = s.propagator(s.inv_direct(mat, postsmooth))
                slv_postsmooth(psi, src)
                t.stop("postsmooth")

                g.message("Done postsmoothing on level %d" % (lvl))

            t.stop("total")

            g.message(
                "Ending inversion routine on level %d: psi = %g, src = %g"
                % (lvl, g.norm2(psi), g.norm2(src))
            )

        return g.matrix_operator(
            mat=invert, otype=otype, zero=(False, False), grid=grid, cb=cb,
        )
