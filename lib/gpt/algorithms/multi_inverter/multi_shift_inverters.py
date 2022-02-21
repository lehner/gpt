#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Mattia Bruno 
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
from gpt.algorithms import base_iterative

# placeholder waiting for new interface  of matrix_operator
# which can handle list of dests
class multi_matrix_operator:
    def __init__(self, n, mat, inv_mat=None):
        self.n = n
        self.mat = mat
        self.inv_mat = inv_mat
        
    def __call__(self, first, second=None):
        if not second is None:
            src = second
            dst = g.core.util.to_list(first)
        if second is None:
            src = first
            if type(src) == g.lattice:
                dst = [g.lattice(src) for  _ in range(self.n)]
            else:
                dst = g.copy(src)
        
        if type(src) == list:
            assert len(src) == self.n
            
        assert len(dst) == self.n
        self.mat(dst, src)
        return dst


class multi_shift_inverter_base(base_iterative):
    def shifted_mat(self, mat, s):
        def operator(dst, src):
            mat(dst, src)
            dst += s * src
        return operator
    
class multi_shift_inverter(multi_shift_inverter_base):
    def __init__(self, base_inverter):
        super().__init__()
        self.base_inverter = base_inverter
            
    def __call__(self, mat, shifts):
        otype, grid, cb = None, None, None
        if type(mat) == g.matrix_operator:
            otype, grid, cb = mat.otype, mat.grid, mat.cb
            mat = mat.mat
            # remove wrapper for performance benefits

        mats = [self.shifted_mat(mat, s) for s in shifts]
        invs = [self.base_inverter(m) for m in mats]
    
        def _mat(dst, src):
            for i, m in enumerate(mats):
                m(dst[i],src[i])
        
        @self.timed_function
        def inv(dst, src, t):
            for i, solver in enumerate(invs):
                dst[i] @= solver * src
            self.log(f"completed")
            
        n = len(shifts)
        return multi_matrix_operator(n, mat=inv, 
                                     inv_mat=multi_matrix_operator(n, _mat))


class shifted_cg:
    def __init__(self, psi, src, s):
        self.s = s
        self.p = g.copy(src)
        self.x = psi
        self.x[:] = 0
        self.a = 1.0
        self.rh = 1.0
        self.gh = 1.0
        self.b = 0.0
        self.converged = False
        
    def step1(self, a, b, om):
        rh = 1.0 / (1.0 + self.s * a + (1.0-self.rh)*om)
        self.rh = rh
        self.a = rh*a
        self.b = rh**2*b
        self.gh = self.gh*rh
        
    def step2(self, r):
        self.x += self.a * self.p
        self.p @= self.b * self.p + self.gh * r
        
    def check(self, cp, rsq):
        if not self.converged:
            if self.gh * cp <= rsq:
                self.converged = True
                return f"shift {self.s} converged"
        return None


class multi_shift_cg(multi_shift_inverter_base):
    @g.params_convention(eps=1e-15, maxiter=1000000)
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]


    def __call__(self, mat, shifts):
        ns = len(shifts)
        
        otype, grid, cb = None, None, None
        if type(mat) == g.matrix_operator:
            otype, grid, cb = mat.otype, mat.grid, mat.cb
            mat = mat.mat
            # remove wrapper for performance benefits
    
        def _mat(dst, src):
            for i, m in enumerate([self.shifted_mat(mat, s) for s in shifts]):
                m(dst[i],src[i])
                            
        @self.timed_function
        def inv(psi, src, t):
            scgs = [shifted_cg(psi[i], src, shifts[i]) for i in range(ns)]
                        
            t("setup")
            p, mmp, r = g.copy(src), g.copy(src), g.copy(src)
            x = g.copy(src)
            x[:] = 0

            b = 0.0
            a = g.norm2(p)
            cp = a
            ssq = g.norm2(src)
            if ssq == 0.0:
                assert a != 0.0  # need either source or psi to not be zero
                ssq = a
            rsq = self.eps ** 2.0 * ssq
            for k in range(self.maxiter):
                c = cp
                t("matrix")
                mat(mmp, p)

                t("inner_product")
                dc = g.inner_product(p, mmp)
                d = dc.real
                om = b/a
                a = c / d
                om *= a

                t("axpy_norm2")
                cp = g.axpy_norm2(r, -a, mmp, r)

                t("linear combination")
                b = cp / c
                for cg in scgs:
                    cg.step1(a, b, om)

                x += a * p
                p @= b * p + r
                for cg in scgs:
                    cg.step2(r)
                
                t("other")
                for cg in scgs:
                    msg = cg.check(cp, rsq)
                    if msg:
                        self.log(f"{msg} in {k+1} iterations")
                if sum([cg.converged for cg in scgs])==ns:
                    return

            self.log(
                f"NOT converged in {k+1} iterations;  squared residual {cp:e} / {rsq:e}"
            )
            
        n = len(shifts)
        return multi_matrix_operator(n, mat=inv, 
                                     inv_mat=multi_matrix_operator(n, _mat))
