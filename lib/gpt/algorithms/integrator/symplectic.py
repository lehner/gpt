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
from gpt.algorithms.integrator import euler


class log:
    def __init__(self):
        self.grad = {}
        self.time = gpt.timer()

    def reset(self):
        self.time = gpt.timer()
        for key in self.grad:
            self.grad[key] = []

    def gradient(self, gs, name):
        if name not in self.grad:
            self.grad[name] = []

        self.time("norm")
        gn = 0.0
        v = 0
        for g in gpt.core.util.to_list(gs):
            gn += gpt.norm2(g)
            v += g.grid.gsites
        self.time()
        self.grad[name].append(gn / v)

    def __call__(self, grad, name):
        def inner():
            self.time(name)
            gs = grad()
            self.time()
            self.gradient(gs, name)
            return gs

        return inner

    def get(self, key):
        return self.grad[key]


def set_verbose(val=True):
    for i in ["leap_frog", "omf2", "omf2_force_gradient", "omf4"]:
        gpt.default.set_verbose(i, val)


class step:
    def __init__(self, funcs, c, n=1):
        self.funcs = gpt.core.util.to_list(funcs)
        self.c = gpt.core.util.to_list(c)
        self.n = n
        self.nf = len(self.funcs)
        if (len(self.c) == 1) and (self.nf > 1):
            self.c = self.c * self.nf

    def __add__(self, s):
        assert self.n == s.n
        return step(self.funcs + s.funcs, self.c + s.c, self.n)

    def __mul__(self, f):
        return step(self.funcs, [c * f for c in self.c], self.n)

    def __call__(self, eps):
        for i in range(self.nf):
            # gpt.message(f"call eps = {eps}, {i} / {self.nf}")
            self.funcs[i](self.c[i] * eps**self.n)


class symplectic_base:
    def __init__(self, N, half, middle, inner, name, tag=None):
        self.N = N
        half = [lambda x: None] if not half else half
        self.cycle = half + middle + list(reversed(half))
        self.inner = gpt.core.util.to_list(inner)
        self.__name__ = f"{name}"
        self.tag = tag
        nc = len(self.cycle)

        self.scheme = []
        if N == 1:
            self.scheme = self.cycle
        else:
            for i in range(N):
                self.scheme += [self.cycle[0] * (2 / N if i > 0 else 1 / N)]
                j = nc if i == N - 1 else nc - 1
                self.scheme += [c * (1 / N) for c in self.cycle[1:j]]

    def string_representation(self, lvl):
        out = f" - Level {lvl} {self.__name__} steps={self.N}"
        for i in self.inner:
            if isinstance(i, symplectic_base):
                out += "\n" + i.string_representation(lvl + 1)
        return out

    def __str__(self):
        return self.string_representation(0)

    def __getitem__(self, args):
        return self.scheme[args]

    def __call__(self, tau):
        eps = tau / self.N
        verbose = gpt.default.is_verbose(self.__name__)

        time = gpt.timer(f"Symplectic integrator {self.__name__} [eps = {eps:.4e}]")
        time(self.__name__)

        for s in self.scheme:
            s(tau)

        if verbose:
            time()
            gpt.message(time)


class update_p(symplectic_base):
    def __init__(self, dst, frc):
        ip = euler(dst, frc, -1)
        super().__init__(1, [], [ip], None, "euler")


class update_q(symplectic_base):
    def __init__(self, dst, frc):
        iq = euler(dst, frc, +1)
        super().__init__(1, [], [iq], None, "euler")


# p1 = 0
# p1 -= d_1 S
# q1  = exp(p1 * a * eps^2) * q0 = exp(-dS * a * eps^2) * q0 = q0  - d_1 S * q0 * a * eps^2 + O(eps^4)
# p  -= d_2 S * b * eps = d_2 { S[q0] - d_1S[q0] * a * eps^2+ O(eps^4) } * b * eps
#    -= d_2 S[q0] * b * eps - d_2 d_1 S[q0] * a * b * eps^3
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

    def __call__(self, a, b):
        scheme = [step(self.init, a / b, 2), step(self.end, b, 1)]

        def inner(eps):
            for s in scheme:
                s(eps)

        return step(inner, 1.0)


# i0: update_momenta, i1: update_dynamical_fields


class leap_frog(symplectic_base):
    def __init__(self, N, i0, i1):
        super().__init__(
            N,
            [step(i0, 0.5) + step(i1[0], 1.0)],
            [step(i1[1:-1], 1.0)],
            i1,
            "leap_frog",
        )


class OMF2(symplectic_base):
    def __init__(self, N, i0, i1, l=0.18):
        r0 = l
        super().__init__(
            N,
            [step(i0, r0) + step(i1[0], 0.5), step(i1[1:-1], 0.5)],
            [step(i0, (1 - 2 * r0)) + step(i1[0], 1.0)],
            i1,
            "omf2",
        )


class OMF2_force_gradient(symplectic_base):
    def __init__(self, N, i0, i1, ifg, l=1.0 / 6.0):
        r0 = l
        middle = [_ifg(2.0 / 72.0, 1 - 2 * r0) for _ifg in gpt.core.util.to_list(ifg)]
        super().__init__(
            N,
            [step(i0, r0) + step(i1[0], 0.5), step(i1[1:-1], 0.5)],
            middle + [step(i1[0], 1.0)],
            i1,
            "omf2_force_gradient",
        )


# Omelyan, Mryglod, Folk, 4th order integrator
#   ''Symplectic analytically integrable decomposition algorithms ...''
#   https://doi.org/10.1016/S0010-4655(02)00754-3
#      values of r's can be found @ page 292, sec 3.5.1, Variant 8
class OMF4(symplectic_base):
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
            [
                step(i0, r[0]) + step(i1[0], r[1]),
                step(i1[1:-1], r[1]),
                step(i0, r[2]) + step(i1[0], r[1] + r[3]),
                step(i1[1:-1], r[3]),
                step(i0, f1) + step(i1[0], r[3] + f2),
            ],
            [step(i1[1:-1], f2)],
            i1,
            "omf4",
        )
