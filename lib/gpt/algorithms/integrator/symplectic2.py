#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020-25  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
        self.verbose = gpt.default.is_verbose("symplectic_log")
        self.last_log = None

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
        if self.verbose:
            if self.last_log is None:
                self.last_log = gpt.time()
            if gpt.time() - self.last_log > 10:
                self.last_log = gpt.time()
                gpt.message("Force status")
                gpt.message(self.time)

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


class symplectic_base:
    def __init__(self, name):
        self.__name__ = name
        self.scheme = []

    def add(self, op, step, direction):
        self.scheme.append((op, step, direction))

    def simplify(self):
        found = False
        prev_scheme = self.scheme
        while True:
            self.scheme = []
            for op, step, direction in prev_scheme:
                if abs(step) > 1e-15:
                    new_scheme = (op, step, 0)
                    if len(self.scheme) > 0 and self.scheme[-1][0] == op:
                        self.scheme[-1] = (op, step + self.scheme[-1][1], 0)
                        found = True
                    else:
                        self.scheme.append(new_scheme)
                else:
                    found = True

            if not found:
                break

            prev_scheme = self.scheme
            found = False

    def __str__(self):
        r = f"{self.__name__}"
        for op, step, direction in self.scheme:
            if isinstance(op, int):
                tag = f"I{op}"
            else:
                assert isinstance(op, tuple)
                tag = op[-1]
            r = r + f"\n  {tag}({step}, {direction})"
        return r

    def insert(self, *ip):
        scheme = []
        for op, step, direction in self.scheme:
            i = ip[op]
            if isinstance(i, symplectic_base):
                for op2, step2, direction2 in i.scheme:
                    scheme.append((op2, step * step2, 0))
            else:
                scheme.append((i, step, 0))
        self.scheme = scheme

    def unwrap(self):
        assert len(self.scheme) == 1
        op, step, direction = self.scheme[0]
        assert step == +1 and direction == 0
        assert isinstance(op, tuple)
        return op[1], op[2]

    def add_directions(self):
        n = len(self.scheme)
        if n % 2 == 1:
            pos = (n - 1) // 2
            if isinstance(self.scheme[pos], tuple) and not self.scheme[pos][0][-2]:
                i, step, direction = self.scheme[pos]
                mid = [(i, step / 2, +1), (i, step / 2, -1)]
                self.scheme = self.scheme[0:pos] + mid + self.scheme[pos + 1 :]

        n = len(self.scheme)
        for pos in range(n // 2):
            if isinstance(self.scheme[pos], tuple) and not self.scheme[pos][0][-2]:
                i, step, direction = self.scheme[pos]
                self.scheme[pos] = (i, step, +1)
                i, step, direction = self.scheme[n - pos - 1]
                self.scheme[n - pos - 1] = (i, step, -1)

    def __call__(self, tau):
        verbose = gpt.default.is_verbose(self.__name__)

        time = gpt.timer(f"Symplectic integrator {self.__name__}")
        time(self.__name__)

        n = len(self.scheme)
        for i in range(n):
            op, step, direction = self.scheme[i]
            if isinstance(op, int):
                raise Exception("Integrator not completely defined")
            else:
                assert isinstance(op, tuple)

                if verbose:
                    gpt.message(
                        f"{self.__name__} on step {i}/{n}: {op[-1]}({step * tau}, {direction})"
                    )

                op[1](step * tau, direction)

        if verbose:
            time()
            gpt.message(time)


def update_variable_general(dst, frc, explicit, tag):
    s = symplectic_base(tag)
    s.add((dst, frc, explicit, tag), +1, 0)
    return s


def update_p(dst, frc, tag="P"):
    ip = euler(dst, frc, -1)
    return update_variable_general(dst, ip, True, tag)


# p1 = 0
# p1 -= d_1 S
# q1  = exp(p1 * a * eps^2) * q0 = exp(-dS * a * eps^2) * q0 = q0  - d_1 S * q0 * a * eps^2 + O(eps^4)
# p  -= d_2 S * b * eps = d_2 { S[q0] - d_1S[q0] * a * eps^2+ O(eps^4) } * b * eps
#    -= d_2 S[q0] * b * eps - d_2 d_1 S[q0] * a * b * eps^3
def update_p_force_gradient(q, iq, p, ip_ex, ip_sl=None, tag="P"):
    q = gpt.util.to_list(q)
    p = gpt.util.to_list(p)

    ip1 = ip_ex if ip_sl is None else ip_sl
    ip2 = ip_ex

    ip1, ip1_explicit = ip1.unwrap()
    ip2, ip2_explicit = ip2.unwrap()
    iq, iq_explicit = iq.unwrap()

    explicit = ip1_explicit
    assert explicit == iq_explicit and explicit == ip2_explicit

    def _create(a, b):

        def _inner(b_eps, direction):

            if explicit:
                assert direction == 0
            else:
                assert direction in [+1, -1]

            if direction in [0, +1]:
                cache_p = gpt.copy(p)
                cache_q = gpt.copy(q)

                for pi in p:
                    pi[:] = 0

                ip1(1.0, direction)
                iq(a / b**3 * b_eps**2, direction)

                gpt.copy(p, cache_p)
                ip2(b_eps, direction)

                gpt.copy(q, cache_q)

            else:
                assert False

            cache_p = None
            cache_q = None

        return update_variable_general(p + q, _inner, explicit, tag)

    return _create


def update_q(dst, frc, tag="Q"):
    ip = euler(dst, frc, +1)
    return update_variable_general(dst, ip, True, tag)


def implicit_update(f, f2, explicit, tag=None, max_iter=100, eps=1e-10):

    if not isinstance(explicit, symplectic_base):

        def _create(*args):
            return implicit_update(f, f2, explicit(*args), tag, max_iter, eps)

        return _create

    f = gpt.util.to_list(f)
    f2 = gpt.util.to_list(f2)
    verbose = gpt.default.is_verbose("implicit_update")

    if tag is None:
        tag = explicit.__name__

    def _update(dt, direction):
        if direction not in [+1, -1]:
            raise Exception(f"Implicit update without well-defined direction {direction}")

        if direction == +1:
            gpt.copy(f2, f)
            return explicit(dt)

        t = gpt.timer("Implicit update")
        t("copy")
        f0 = gpt.copy(f)
        for i in range(max_iter):
            gpt.copy(f2, f)
            gpt.copy(f, f0)
            t(f"iteration {i}")
            explicit(dt)
            t()
            resid = 0.0
            for j in range(len(f)):
                resid += gpt.norm2(f2[j] - f[j]) / f[j].grid.gsites
            resid = resid**0.5

            if verbose:
                gpt.message(f"Implicit update step {i}: {resid:e} / {eps:e}")

            if resid < eps:
                if verbose:
                    gpt.message(t)
                break

    return update_variable_general(f + f2, _update, False, tag)


def complete_coefficients(r):
    x0 = sum(r[0::2])
    x1 = sum(r[1::2])
    r.append(0.5 - x0)
    r.append(0.5 - x1)


def force_general(N, ia, r, q, tag):
    s = symplectic_base(f"{tag}({N}, {ia[0].__name__}, {ia[1].__name__})")

    for j in range(N):
        for i in range(len(r) // 2):
            s.add(q[2 * i + 0], r[2 * i + 0] / N, 0)
            s.add(q[2 * i + 1], r[2 * i + 1] / N, 0)

        for i in reversed(range(len(r) // 2)):
            s.add(q[2 * i + 1], r[2 * i + 1] / N, 0)
            s.add(q[2 * i + 0], r[2 * i + 0] / N, 0)

    s.simplify()

    s.insert(*ia)

    s.simplify()

    s.add_directions()

    return s


# force-gradient integrators
def OMF2_force_gradient(N, i0, i1, ifg, l=1.0 / 6.0):
    # https://arxiv.org/pdf/0910.2950
    r = [l, 0.5]
    q = [0, 1, 2, 3]  # 3 is never used in this scheme
    complete_coefficients(r)
    ifg = ifg(2.0 / 72.0, 2.0 * r[-2])
    return force_general(N, [i0, i1, ifg], r, q, "OMF2_force_gradient")


# force integrators
def OMF4(N, i0, i1):
    # Omelyan, Mryglod, Folk, 4th order integrator
    #   ''Symplectic analytically integrable decomposition algorithms ...''
    #   https://doi.org/10.1016/S0010-4655(02)00754-3
    #      values of r's can be found @ page 292, sec 3.5.1, Variant 8
    r = [
        0.08398315262876693,
        0.2539785108410595,
        0.6822365335719091,
        -0.03230286765269967,
    ]
    complete_coefficients(r)
    q = [0, 1, 0, 1, 0, 1]
    return force_general(N, [i0, i1], r, q, "OMF4")


def OMF2(N, i0, i1, l=0.18):
    r = [l, 0.5]
    complete_coefficients(r)
    q = [0, 1, 0, 1]
    return force_general(N, [i0, i1], r, q, "OMF2")


def leap_frog(N, i0, i1):
    return force_general(N, [i0, i1], [0.5, 0.5], [0, 1], "leap_frog")
