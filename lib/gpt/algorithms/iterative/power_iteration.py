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
import gpt as g

class power_iteration:

    @g.params_convention()
    def __init__(self, params):
        self.params = params
        self.tol = params["eps"]
        self.maxit = params["maxiter"]

    def __call__(self,mat,src):
        verbose=g.default.is_verbose("power_iteration")

        dst,tmp=g.lattice(src),g.copy(src)

        tmp /= g.norm2(tmp)**0.5

        ev_prev=None
        for it in range(self.maxit):
            mat(dst,tmp)
            ev=g.norm2(dst)**0.5
            if verbose:
                g.message("eval_max[ %d ] = %g" % (it,ev))
            tmp @= (1.0/ev) * dst
            if not ev_prev is None:
                if abs(ev - ev_prev) < self.tol*ev:
                    if verbose:
                        g.message("Converged")
                    return (ev,tmp,True)
            ev_prev=ev

        return (ev,tmp,False)
