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


class simple:
    def __init__(self):
        self.t_setup = g.timer()
        self.t_solve = g.timer()

    def __call__(self, matrix=None):
        otype, grid, cb = None, None, None
        if type(matrix) == g.matrix_operator:
            otype, grid, cb = matrix.otype, matrix.grid, matrix.cb
            matrix = matrix.mat

        def inv(psi, src):
            self.t_solve.start("total")
            assert src != psi
            psi @= src
            self.t_solve.stop("total")

        return g.matrix_operator(mat=inv, inv_mat=matrix, otype=otype, grid=grid, cb=cb)


class mg:
    def __init__(self, mat_f, params):
        self.params = params
        self.levels = params["levels"]
        self.grid_f = params["grid_f"]
        self.grid_c = params["grid_c"]
        self.northo = params["northo"]
        self.nbasis = params["nbasis"]
        self.hermitian = params["hermitian"]

        self.level = 0

        assert self.level == 0
        assert self.nbasis % 2 == 0
        self.t_setup = g.timer()
        self.t_solve = g.timer()

        self.nb = self.nbasis // 2

        # timing
        self.t_setup.start("misc")

        # store fine matrix
        self.mat_f = mat_f

        # abbreviations
        s = g.qcd.fermion.solver
        a = g.algorithms.iterative

        # fine algorithms / solvers
        slv_alg_c = a.fgcr({"eps": 1e-1, "maxiter": 100, "restartlen": 20})
        presmth_alg_f = a.fgmres({"eps": 1e-2, "maxiter": 16, "restartlen": 16})
        # postsmth_alg_f = a.fgmres({"eps": 1e-1, "maxiter": 16, "restartlen": 4})
        postsmth_alg_f = a.mr({"eps": 1e-1, "maxiter": 16, "relax": 1})
        setup_alg_f = a.fgmres({"eps": 1e-1, "maxiter": 16, "restartlen": 16})
        self.slv_presmth_f = s.propagator(s.inv_direct(self.mat_f, presmth_alg_f))
        self.slv_postsmth_f = s.propagator(s.inv_direct(self.mat_f, postsmth_alg_f))
        self.slv_setup_f = s.propagator(s.inv_direct(self.mat_f, setup_alg_f))
        self.t_setup.stop("misc")

        # trying around with setup solver
        eo2_odd = g.qcd.fermion.preconditioner.eo2(self.mat_f, parity=g.odd)
        setup_alg_f = a.cg({"eps": 1e-5, "maxiter": 100})
        self.slv_setup_f = s.propagator(s.inv_eo_ne(eo2_odd, setup_alg_f))

        # create basis / intergrid operator
        g.message("Starting creation of null-space vectors")
        self.t_setup.start("setup_vecs")
        if self.level == 0:
            self.basis = [g.vspincolor(self.grid_f) for __ in range(self.nbasis)]
        self.rng = g.random("multigrid")
        self.rng.cnormal(self.basis)
        self.t_setup.stop("setup_vecs")

        # create near null vectors
        self.t_setup.start("find_null_space")
        src, psi = g.copy(self.basis[0]), g.copy(self.basis[0])
        for i, v in enumerate(self.basis[0 : self.nb]):
            psi[:] = 0
            src @= v
            self.slv_setup_f(psi, src)
            v @= psi
        # g.orthonormalize(self.basis)
        self.t_setup.stop("find_null_space")
        g.message("Done creating null-space vectors")

        self.t_setup.start("ortho_vecs")
        g.split_chiral(self.basis)
        for i in range(self.northo):
            g.message("Block ortho step %d" % i)
            g.block.orthonormalize(self.grid_c, self.basis)
        self.t_setup.stop("ortho_vecs")

        # create coarse links + operator
        self.t_setup.start("create_coarse")
        self.A = [g.mcomplex(self.grid_c, self.nbasis) for __ in range(9)]
        g.coarse.create_links(self.A, mat_f, self.basis)
        self.mat_c = g.qcd.fermion.coarse(
            self.A, {"hermitian": self.hermitian, "level": self.level}
        )
        self.t_setup.stop("create_coarse")
        g.message("Done setting up")

        # mg operators
        self.slv_c = s.propagator(s.inv_direct(self.mat_c, slv_alg_c))
        self.f2c = g.block.project
        self.c2f = g.block.promote

        self.t_setup.print("mg_setup")

    def __call__(self, matrix=None):
        # ignore matrix
        otype, grid, cb = None, None, None
        if type(self.mat_f) == g.matrix_operator:
            otype, grid, cb = self.mat_f.otype, self.mat_f.grid, self.mat_f.cb
            self.mat_f = self.mat_f.mat

        # def inv(psi, src):
        #     assert src != psi
        #     psi @= src

        def inv(psi, src):
            self.t_solve.start("misc")

            src2 = g.norm2(src)

            g.message("mg.vcycle, input: norm2(psi) = %g" % (g.norm2(psi)))

            assert src != psi
            r = g.copy(src)
            r_c, e_c = (
                g.vcomplex(self.grid_c, self.nbasis),
                g.vcomplex(self.grid_c, self.nbasis),
            )
            e_c[:] = 0
            uiae = g.copy(src)
            self.t_solve.stop("misc")

            # optional pre-smoothing
            self.t_solve.start("presmoooth")
            if False:
                tmp, mmtmp = gpt.lattice(src), gpt.lattice(src)
                tmp[:] = 0
                self.pre_smth_f(tmp, src)
                self.mat_f.M(mmtmp, tmp)
                r @= src - mmtmp
            else:
                r @= src
            self.t_solve.stop("presmoooth")

            # fine to coarse
            self.t_solve.start("fine2coarse")
            self.f2c(r_c, r, self.basis)
            self.t_solve.stop("fine2coarse")

            # coarse solve
            self.t_solve.start("coarse_solve")
            g.message("Solve on coarse level: start")
            g.message(
                "Before coarse solve: norm2(e_c) = %g, norm2(r_c) = %g"
                % (g.norm2(e_c), g.norm2(r_c))
            )
            self.slv_c(e_c, r_c)
            g.message("Solve on coarse level: stop")
            self.t_solve.stop("coarse_solve")

            # coarse to fine
            self.t_solve.start("coarse2fine")
            self.c2f(e_c, psi, self.basis)
            g.message("mg.vcycle, afterc2f: norm2(psi) = %g" % (g.norm2(psi)))
            self.t_solve.stop("coarse2fine")

            self.mat_f.M(uiae, psi)
            uiae @= src - uiae
            r2_cgc = g.norm2(uiae) / src2

            # post-smoothing (psi as starting guess)
            self.t_solve.start("postsmooth")
            g.message("Post-Smoothing on fine level: start")
            self.slv_postsmth_f(psi, src)
            g.message("mg.vcycle, afterc2f: norm2(psi) = %g" % (g.norm2(psi)))
            g.message("Post-Smoothing on fine level: stop")
            self.t_solve.stop("postsmooth")

            self.mat_f.M(uiae, psi)
            uiae @= src - uiae
            r2_postsm = g.norm2(uiae) / src2

            g.message(
                "input norm = %g, coarse res = %g, post-smooth res = %g"
                % (src2, r2_cgc, r2_postsm)
            )

        return g.matrix_operator(
            mat=inv, otype=otype, zero=(False, False), grid=grid, cb=cb,
        )
