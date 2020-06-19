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
import cgpt, gpt, numpy
from gpt.params import params_convention

class covariant_shift:
    @params_convention()
    def __init__(self, U, params):
        self.nd = len(U)
        self.L = U[0].grid.fdimensions
        self.U = [ gpt.copy(u) for u in U ]
        self.params = params

        for mu in range(self.nd):
            last_slice=tuple([ self.L[mu]-1 if mu == nu else slice(None,None,None) for nu in range(self.nd) ])
            self.U[mu][last_slice]=self.U[mu][last_slice] * self.params["boundary_phases"][mu]

        # now take boundary_phase from params and apply here
        self.Udag = [ gpt.eval(gpt.adj(u)) for u in self.U ]

        def _forward(mu):
            def wrap(dst, src):
                dst @= self.U[mu]*gpt.cshift(src,mu,+1)
            return wrap

        def _backward(mu):
            def wrap(dst, src):
                dst @= gpt.cshift(self.Udag[mu]*src,mu,-1)
            return wrap

        self.forward = [ gpt.matrix_operator(mat = _forward(mu), inv_mat = _backward(mu)) for mu in range(self.nd) ]
        self.backward = [ gpt.matrix_operator(inv_mat = _forward(mu), mat = _backward(mu)) for mu in range(self.nd) ]

