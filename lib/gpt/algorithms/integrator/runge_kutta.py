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


def left_increment(dst, src_left, scale):
    dst = g.util.to_list(dst)
    src_left = g.util.to_list(src_left)
    group = dst[0].otype
    algebra = group.cartesian()
    assert src_left[0].otype.__name__ == algebra.__name__

    for src_left_mu, dst_mu in zip(src_left, dst):
        if group.__name__ != algebra.__name__:
            dst_mu @= group.compose(
                g.project(g.convert(g(scale * src_left_mu), group), "defect"), dst_mu
            )
        else:
            dst_mu @= group.compose(g(scale * src_left_mu), dst_mu)


def runge_kutta(src, d_src_cartesian, epsilon, code):
    dst = g.copy(src)
    z = d_src_cartesian(dst)
    for uf, zf in code:
        left_increment(dst, z, uf * epsilon)
        if zf is not None:
            left_increment(z, d_src_cartesian(dst), zf)
    return dst


def runge_kutta_4(src, d_src_cartesian, epsilon):
    # Generalization of (C.2) of https://arxiv.org/pdf/1006.4518.pdf
    # to integrate also additive groups with same interface
    #
    # Example: group = (\mathbb{C},+)
    #
    #  src(t+eps) = src(t) + eps * d_src_cartesian(src(t)) + O(eps^2)
    #  Notation: src'(t) = d_src_cartesian(src(t))
    #
    # Example: group = SU(n), U(1)
    #
    #  src(t+eps) = exp(i*d_src_cartesian(src(t))*eps)*src(t) + O(eps^2)
    #  lim_{eps -> 0} (src(t+eps) - src(t)) / eps = i*d_src_cartesian
    #  Notation: src'(t) = exp(i*d_src_cartesian(src(t)))*src(t)
    #
    # Input:  src(t)
    # Output: src(t+eps)
    code = [(1.0 / 4.0, -32.0 / 17.0), (-17.0 / 36.0, 27.0 / 17.0), (17.0 / 36.0, None)]
    return runge_kutta(src, d_src_cartesian, epsilon, code)
