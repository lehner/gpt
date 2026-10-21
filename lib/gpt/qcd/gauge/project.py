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
def _fused(*fields):
    # the projections as one site-local program (g.local_program): plain N x N
    # matrix fields in one memory block on a full grid
    return all(
        isinstance(x, g.lattice)
        and len(x.v_obj) == 1
        and len(x.otype.shape) == 2
        and x.otype.shape[0] == x.otype.shape[1]
        and x.grid.cb.n == 1
        for x in fields
    )


def _traceless(h, sign):
    # the traceless part of (h + sign h^dag) / 2 (symbolic)
    N = h.n
    x = h * 0.5 + h.adj() * (0.5 * sign)
    return x - g.local_program.matrix.identity(N) * (x.trace() * (1.0 / N))


_programs = {}


def _program(name, otype, output=None):
    # traceless_(anti_)hermitian of a field of type otype, or with name
    # "infinitesimal_to_cartesian" the SU(N) conversion TH(dU U^dag / 2i) of
    # (dU, U), as one kernel (output: the otype of the result)
    key = (name, otype.__name__)
    if key not in _programs:
        if name == "infinitesimal_to_cartesian":
            _programs[key] = g.local_program(
                lambda dU, U: _traceless((dU * U.adj()) * (1.0 / 2j), 1.0),
                inputs=[otype, otype],
                outputs=[output],
            )
        else:
            sign = 1.0 if name == "traceless_hermitian" else -1.0
            _programs[key] = g.local_program(lambda x: _traceless(x, sign), inputs=[otype], outputs=[otype])
    return _programs[key]


def traceless_anti_hermitian(src):
    if isinstance(src, list):
        return [traceless_anti_hermitian(x) for x in src]
    if isinstance(src, g.expr):
        src = g.eval(src)
    foundation = src.__class__.foundation
    if hasattr(foundation, "traceless_anti_hermitian"):
        return foundation.traceless_anti_hermitian(src)
    if _fused(src):
        return _program("traceless_anti_hermitian", src.otype)(src)[0]
    N = src.otype.shape[0]
    ret = g(0.5 * src - 0.5 * g.adj(src))
    ret -= g.identity_constant(src) * g.trace(ret) / N
    return ret


def traceless_hermitian(src):
    if isinstance(src, list):
        return [traceless_hermitian(x) for x in src]

    src = g.eval(src)
    foundation = src.__class__.foundation
    if hasattr(foundation, "traceless_hermitian"):
        return foundation.traceless_hermitian(src)
    if _fused(src):
        return _program("traceless_hermitian", src.otype)(src)[0]
    N = src.otype.shape[0]
    ret = g(0.5 * src + 0.5 * g.adj(src))
    ret -= g.identity_constant(src) * g.trace(ret) / N
    return ret
