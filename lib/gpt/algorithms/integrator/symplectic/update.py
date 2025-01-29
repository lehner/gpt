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
from gpt.algorithms.integrator.symplectic import symplectic_base


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
