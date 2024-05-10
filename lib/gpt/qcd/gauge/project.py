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


# (2) of https://arxiv.org/pdf/hep-lat/0311018.pdf
def traceless_anti_hermitian(src):
    if isinstance(src, list):
        return [traceless_anti_hermitian(x) for x in src]
    if isinstance(src, g.expr):
        src = g.eval(src)
    N = src.otype.shape[0]
    ret = g(0.5 * src - 0.5 * g.adj(src))
    ret -= g.identity(src) * g.trace(ret) / N
    return ret


def traceless_hermitian(src):
    if isinstance(src, list):
        return [traceless_hermitian(x) for x in src]

    src = g.eval(src)
    N = src.otype.shape[0]
    ret = g(0.5 * src + 0.5 * g.adj(src))
    ret -= g.identity(src) * g.trace(ret) / N
    return ret
