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


def multi_caller(funcs, arg):
    for f in funcs:
        f(arg)


# i0: update_momenta, i1: update_dynamical_fields
class integrator_base:
    def __init__(self, N, i0, i1, coeffs, name):
        self.N = N
        self.i0 = gpt.core.util.to_list(i0)
        self.i1 = gpt.core.util.to_list(i1)

        self.coeffs = [coeffs[0]]
        for i in range(N - 1):
            self.coeffs.extend(coeffs[1:-1])
            self.coeffs += [2.0 * coeffs[-1]]
        self.coeffs.extend(coeffs[1:])

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

        multi_caller(self.i0, eps * self.coeffs[0])
        k = 1
        while k < len(self.coeffs):
            multi_caller(self.i1, eps * self.coeffs[k])
            k += 1
            multi_caller(self.i0, eps * self.coeffs[k])
            k += 1

        if verbose:
            time()
            gpt.message(f"{self.__name__} Integrator ran in {time.dt['total']:g} secs")


class leap_frog(integrator_base):
    def __init__(self, N, i0, i1):
        super().__init__(N, i0, i1, [0.5, 1.0, 0.5], "leap_frog")


class OMF2(integrator_base):
    def __init__(self, N, i0, i1, l=0.18):
        r0 = l
        super().__init__(N, i0, i1, [r0, 0.5, (1 - 2 * r0), 0.5, r0], "omf2")


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
        super().__init__(
            N,
            i0,
            i1,
            [r[0], r[1], r[2], r[3], f1, f2, f1, r[3], r[2], r[1], r[0]],
            "omf4",
        )
