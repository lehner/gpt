#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020 Tilo Wettig
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
import numpy as np
from gpt.params import params_convention


def plaquette(U):
    # U[mu](x)*U[nu](x+mu)*adj(U[mu](x+nu))*adj(U[nu](x))
    tr = 0.0
    vol = float(U[0].grid.fsites)
    Nd = len(U)
    ndim = U[0].otype.shape[0]
    for mu in range(Nd):
        for nu in range(mu):
            tr += g.sum(
                g.trace(
                    U[mu]
                    * g.cshift(U[nu], mu, 1)
                    * g.adj(g.cshift(U[mu], nu, 1))
                    * g.adj(U[nu])
                )
            )
    return 2.0 * tr.real / vol / Nd / (Nd - 1) / ndim


def fund2adj(U):
    """
    For SU(2), convert fundamental to adjoint representation

    Input: fundamental gauge field

    Output: adjoint gauge field
    """
    if type(U) == list:
        return [fund2adj(x) for x in U]

    assert type(U) == g.lattice, "Input must be lattice object"
    assert (
        U[0, 0, 0, 0].otype.__name__ == "ot_matrix_su2_fundamental()"
    ), "Input gauge field must be SU(2) fundamental"

    grid = U.grid
    T = U.otype.generators(grid.precision.complex_dtype)
    V_idx = {}
    for a in range(len(T)):
        for b in range(len(T)):
            V_idx[a, b] = g.eval(2.0 * g.trace(T[a] * U * T[b] * g.adj(U)))

    V = g.lattice(g.matrix_su2_adjoint(grid))
    g.merge_color(V, V_idx)
    return V


@params_convention(otype=None, Nd=None)
def create_links(first, init, params):
    if type(first) == g.grid:

        # default representation is SU3 fundamental
        if params["otype"] is None:
            params["otype"] = g.ot_matrix_su3_fundamental()

        # default dimension is four
        if params["Nd"] is None:
            params["Nd"] = 4

        # create lattices
        U = [g.lattice(first, params["otype"]) for i in range(params["Nd"])]

        # initialize them
        create_links(U, init, params)
        return U

    elif type(first) == list:

        # if given a list, the dimensionality can be inferred
        if params["Nd"] is None:
            params["Nd"] = len(first)
        else:
            assert params["Nd"] == len(first)

        # initialize each link
        for x in first:
            create_links(x, init, params)
        return first

    elif type(first) == g.lattice:

        # if otype is given, make sure it is expected
        if params["otype"] is not None:
            assert params["otype"].__name__ == first.otype.__name__

        init(first, params)
        return first

    else:
        assert 0


@params_convention(scale=1.0)
def random(first, rng, params):
    def init(x, p):
        rng.lie(x, scale=p["scale"])

    return create_links(first, init, params)


@params_convention()
def unit(first, params):
    def init(x, p):
        otype = x.otype
        x[:] = g.gpt_object(
            np.identity(otype.shape[0], dtype=x.grid.precision.complex_dtype), otype
        )

    return create_links(first, init, params)
