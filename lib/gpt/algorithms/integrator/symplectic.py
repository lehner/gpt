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

import gpt


def update_p(dst, frc):
    return update(dst, frc, -1)


def update_q(dst, frc):
    return update(dst, frc, +1)


def update(dst, frc, sign):
    dst = gpt.core.util.to_list(dst)
    force = frc

    def micro(eps):
        forces = gpt.core.util.to_list(force())
        for i in range(len(forces)):
            dst[i] @= gpt.group.compose(gpt.eval(sign * eps * forces[i]), dst[i])

    return micro


def step(funcs, c, n=1):
    funcs = gpt.core.util.to_list(funcs)
    def inner(eps):
        for f in funcs:
            f(c * eps**n)
    return inner


class integrator_base:
    def __init__(self, N, first_half, middle, name):
        self.N = N
        self.scheme = first_half + middle + list(reversed(first_half))
        self.__name__ = f"{name}({N})"
        
    def string_representation(self, lvl):
        out = f" - Level {lvl} = {self.__name__}"
        for i in self.i1:
            if isinstance(i, integrator_base):
                out += "\n" + i.string_repr(lvl + 1)
        return out

    def __str__(self):
        return self.string_representation(0)


    def __call__(self, tau):
        eps = tau / self.N
        verbose = gpt.default.is_verbose(self.__name__)

        time = gpt.timer(self.__name__)
        time(self.__name__)
        
        self.scheme[0](eps)
        for i in range(self.N-1):
            for s in self.scheme[1:-1]:
                s(eps)
            self.scheme[-1](2*eps)
        for s in self.scheme[1:]:
            s(eps)
            
        if verbose:
            time()
            gpt.message(f"{self.__name__} Integrator ran in {time.dt['total']:g} secs")


# p1 = 0
# p1 -= d_1 S
# q1  = exp(p1 * a) * q0 = exp(-dS * a) * q0 = q0  - d_1 S * q0 * a + O(a^2)
# p  -= d_2 S * b = d_2 { S[q0] - d_1S[q0] * a + O(a^2) } * b 
#    -= d_2 S[q0] * b - d_2 d_1 S[q0] * a * b
class update_p_force_gradient:
    def __init__(self, q, iq, p, ip_ex, ip_sl=None):
        self.q = q
        self.p = p
        self.ip1 = ip_ex if ip_sl is None else ip_sl
        self.iq = iq
        self.ip2 = ip_ex
        self.cache_p = None
        self.cache_q = None
        
    def init(self, arg):
        self.cache_p = gpt.copy(self.p)
        self.cache_q = gpt.copy(self.q)
        for p in gpt.core.util.to_list(self.p):
            p[:] = 0
        self.ip1(1.0)
        self.iq(arg)
        
    def end(self, arg):
        gpt.copy(self.p, self.cache_p)
        self.ip2(arg)
        gpt.copy(self.q, self.cache_q)
        self.cache_p = None
        self.cache_q = None


# i0: update_momenta, i1: update_dynamical_fields

class leap_frog(integrator_base):
    def __init__(self, N, i0, i1):
        super().__init__(N, [step(i0, 0.5)], [step(i1, 1.0)], "leap_frog")
        

class OMF2(integrator_base):
    def __init__(self, N, i0, i1, l=0.18):
        r0 = l
        super().__init__(N, [step(i0, r0), step(i1, 0.5)], [step(i0, (1 - 2 * r0))], "omf2")


class OMF2_force_gradient(integrator_base):
    def __init__(self, N, i0, i1, ifg, l=1./6.):
        r0 = l
        super().__init__(N, [step(i0, r0), step(i1, 0.5)], [
            step(ifg.init, 2./72./(1-2*r0), 2), 
            step(ifg.end, (1-2*r0), 1)
        ], "omf2_force_gradient")


# Omelyan, Mryglod, Folk, 4th order integrator
#   ''Symplectic analytically integrable decomposition algorithms ...''
#   https://doi.org/10.1016/S0010-4655(02)00754-3
#      values of r's can be found @ page 292, sec 3.5.1, Variant 8
class OMF4(integrator_base):
    def __init__(self, N, i0, i1):
        r = [
            0.08398315262876693,
            0.2539785108410595,
            0.6822365335719091,
            -0.03230286765269967,
        ]
        f1 = 0.5 - r[0] - r[2]
        f2 = 1.0 - 2.0 * (r[1] + r[3])
        super().__init__(N, [step(i0, r[0]), step(i1, r[1]), step(i0, r[2]),step(i1, r[3]),step(i0, f1)],[step(i1, f2)], "omf4")
