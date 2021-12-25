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
import numpy as np
from gpt.params import params_convention


@params_convention(otype=None, Nd=None, scale=None)
def create_links(first, init, params):
    if type(first) == g.grid:

        # default representation is SU3 fundamental
        if params["otype"] is None:
            params["otype"] = g.ot_matrix_su_n_fundamental_group(3)

        # default dimension is grid's dimension
        if params["Nd"] is None:
            params["Nd"] = first.nd

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
        rng.element(x, scale=p["scale"])

    return create_links(first, init, params)


@params_convention(Nd=None)
def unit(first, params):
    def init(x, p):
        otype = x.otype
        x[:] = g.gpt_object(
            np.identity(otype.shape[0], dtype=x.grid.precision.complex_dtype), otype
        )

    return create_links(first, init, params)
