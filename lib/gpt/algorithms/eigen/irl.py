#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
#          The history of this version of the algorithm is long.  It is mostly based on
#          my earlier version at https://github.com/lehner/Grid/blob/legacy/lib/algorithms/iterative/ImplicitlyRestartedLanczos.h
#          which is based on code from Tom Blum, Taku Izubuchi, Chulwoo Jung and guided by the PhD thesis of Rudy Arthur.
#          I also adopted some of Peter Boyle's convergence test modifications that are in https://github.com/paboyle/Grid .
#
import gpt as g
import numpy as np
import sys


# Implicitly Restarted Lanczos
class irl:
    @g.params_convention(
        orthogonalize_nblock=4,
        mem_report=False,
        rotate_use_accelerator=True,
        Nm=None,
        Nk=None,
        Nstop=None,
        resid=None,
        betastp=None,
        maxiter=None,
        Nminres=None,
    )
    def __init__(self, params):
        self.params = params
        self.napply = 0

    def __call__(self, mat, src, ckpt=None):
        # verbosity
        verbose = g.default.is_verbose("irl")

        # checkpointer
        if ckpt is None:
            ckpt = g.checkpointer_none()
        ckpt.grid = src.grid
        self.ckpt = ckpt

        # first approximate largest eigenvalue
        g.default.push_verbose("power_iteration_convergence", True)
        pit = g.algorithms.eigen.power_iteration(eps=0.02, maxiter=10, real=True)
        g.default.pop_verbose()
        lambda_max = pit(mat, src)[0]

        # parameters
        Nm = self.params["Nm"]
        Nk = self.params["Nk"]
        Nstop = self.params["Nstop"]
        rotate_use_accelerator = self.params["rotate_use_accelerator"]
        assert Nm >= Nk and Nstop <= Nk

        # tensors
        dtype = np.float64
        lme = np.empty((Nm,), dtype)
        lme2 = np.empty((Nm,), dtype)
        ev = np.empty((Nm,), dtype)
        ev2 = np.empty((Nm,), dtype)
        ev2_copy = np.empty((Nm,), dtype)

        # fields
        f = g.lattice(src)
        v = g.lattice(src)
        evec = [g.lattice(src) for i in range(Nm)]

        # scalars
        k1 = 1
        k2 = Nk
        beta_k = 0.0

        # set initial vector
        evec[0] @= src / g.norm2(src) ** 0.5

        # initial Nk steps
        for k in range(Nk):
            self.step(mat, ev, lme, evec, f, Nm, k)

        # restarting loop
        for it in range(self.params["maxiter"]):
            if verbose:
                g.message("Restart iteration %d" % it)
            for k in range(Nk, Nm):
                self.step(mat, ev, lme, evec, f, Nm, k)
            f *= lme[Nm - 1]

            # eigenvalues
            for k in range(Nm):
                ev2[k] = ev[k + k1 - 1]
                lme2[k] = lme[k + k1 - 1]

            # diagonalize
            t0 = g.time()
            Qt = np.identity(Nm, dtype)
            self.diagonalize(ev2, lme2, Nm, Qt)
            t1 = g.time()

            if verbose:
                g.message("Diagonalization took %g s" % (t1 - t0))

            # sort
            ev2_copy = ev2.copy()
            ev2 = list(reversed(sorted(ev2)))

            # implicitly shifted QR transformations
            Qt = np.identity(Nm, dtype)
            t0 = g.time()
            for ip in range(k2, Nm):
                g.qr_decomposition(ev, lme, Nm, Nm, Qt, ev2[ip], k1, Nm)
            t1 = g.time()

            if verbose:
                g.message("QR took %g s" % (t1 - t0))

            # rotate
            t0 = g.time()
            g.rotate(evec, Qt, k1 - 1, k2 + 1, 0, Nm, rotate_use_accelerator)
            t1 = g.time()

            if verbose:
                g.message("Basis rotation took %g s" % (t1 - t0))

            # compression
            f *= Qt[k2 - 1, Nm - 1]
            f += lme[k2 - 1] * evec[k2]
            beta_k = g.norm2(f) ** 0.5
            betar = 1.0 / beta_k
            evec[k2] @= betar * f
            lme[k2 - 1] = beta_k

            if verbose:
                g.message("beta_k = ", beta_k)

            # convergence test
            if it >= self.params["Nminres"]:
                if verbose:
                    g.message("Rotation to test convergence")

                # diagonalize
                for k in range(Nm):
                    ev2[k] = ev[k]
                    lme2[k] = lme[k]
                Qt = np.identity(Nm, dtype)

                t0 = g.time()
                self.diagonalize(ev2, lme2, Nk, Qt)
                t1 = g.time()

                if verbose:
                    g.message("Diagonalization took %g s" % (t1 - t0))

                B = g.copy(evec[0])

                allconv = True
                if beta_k >= self.params["betastp"]:
                    jj = 1
                    while jj <= Nstop:
                        j = Nstop - jj
                        g.linear_combination(B, evec[0:Nk], Qt[j, 0:Nk])
                        B *= 1.0 / g.norm2(B) ** 0.5
                        if not ckpt.load(v):
                            mat(v, B)
                            ckpt.save(v)
                        ev_test = g.inner_product(B, v).real
                        eps2 = g.norm2(v - ev_test * B) / lambda_max**2.0
                        if verbose:
                            g.message(
                                "%-65s %-45s %-50s"
                                % (
                                    "ev[ %d ] = %s" % (j, ev2_copy[j]),
                                    "<B|M|B> = %s" % (ev_test),
                                    "|M B - ev B|^2 / ev_max^2 = %s" % (eps2),
                                )
                            )
                        if eps2 > self.params["resid"]:
                            allconv = False
                        if jj == Nstop:
                            break
                        jj = min([Nstop, 2 * jj])

                if allconv:
                    if verbose:
                        g.message("Converged in %d iterations" % it)
                        break

        t0 = g.time()
        g.rotate(evec, Qt, 0, Nstop, 0, Nk, rotate_use_accelerator)
        t1 = g.time()

        if verbose:
            g.message("Final basis rotation took %g s" % (t1 - t0))

        return (evec[0:Nstop], ev2_copy[0:Nstop])

    def diagonalize(self, lmd, lme, Nk, Qt):
        TriDiag = np.zeros((Nk, Nk), dtype=Qt.dtype)
        for i in range(Nk):
            TriDiag[i, i] = lmd[i]
        for i in range(Nk - 1):
            TriDiag[i, i + 1] = lme[i]
            TriDiag[i + 1, i] = lme[i]
        w, v = np.linalg.eigh(TriDiag)
        for i in range(Nk):
            lmd[Nk - 1 - i] = w[i]
            for j in range(Nk):
                Qt[Nk - 1 - i, j] = v[j, i]

    def step(self, mat, lmd, lme, evec, w, Nm, k):
        assert k < Nm

        verbose = g.default.is_verbose("irl")
        ckpt = self.ckpt

        alph = 0.0
        beta = 0.0

        evec_k = evec[k]

        results = [w, alph, beta]
        if ckpt.load(results):
            w, alph, beta = results  # use checkpoint

            if verbose:
                g.message(
                    "%-65s %-45s" % ("alpha[ %d ] = %s" % (k, alph), "beta[ %d ] = %s" % (k, beta))
                )

        else:
            if self.params["mem_report"]:
                g.mem_report(details=False)

            # compute
            t0 = g.time()
            mat(w, evec_k)
            t1 = g.time()

            # allow to restrict maximal number of applications within run
            self.napply += 1
            if "maxapply" in self.params:
                if self.napply == self.params["maxapply"]:
                    if verbose:
                        g.message("Maximal number of matrix applications reached")
                    sys.exit(0)

            if k > 0:
                w -= lme[k - 1] * evec[k - 1]

            zalph = g.inner_product(evec_k, w)
            alph = zalph.real

            w -= alph * evec_k

            beta = g.norm2(w) ** 0.5
            w /= beta

            t2 = g.time()
            if k > 0:
                g.orthogonalize(w, evec[0:k], nblock=self.params["orthogonalize_nblock"])
            t3 = g.time()

            ckpt.save([w, alph, beta])

            if verbose:
                g.message(
                    "%-65s %-45s %-50s"
                    % (
                        "alpha[ %d ] = %s" % (k, zalph),
                        "beta[ %d ] = %s" % (k, beta),
                        " timing: %g s (matrix), %g s (ortho)" % (t1 - t0, t3 - t2),
                    )
                )

        lmd[k] = alph
        lme[k] = beta

        if k < Nm - 1:
            evec[k + 1] @= w
