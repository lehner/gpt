#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


def is_field(x):
    if isinstance(x, g.lattice):
        return True
    elif isinstance(x, g.tensor):
        return False
    elif isinstance(x, g.expr):
        return x.lattice() is not None
    elif g.util.is_num(x):
        return False
    elif isinstance(x, g.ad.forward.series):
        for t in x.terms:
            return is_field(x[t])
        raise Exception("Empty series")
    else:
        raise Exception(f"Unknown object type {type(x)}")


def accumulate_gradient(lhs, rhs_gradient, getter=None, setter=None):
    lhs_gradient = lhs.gradient
    if getter is not None:
        lhs_gradient = getter(lhs_gradient)
    rhs_field = is_field(rhs_gradient)
    lhs_field = is_field(lhs_gradient)
    if rhs_field and not lhs_field:
        rhs_gradient = g.sum(rhs_gradient)
    if g.util.is_num(lhs_gradient) and isinstance(rhs_gradient, g.expr):
        rhs_gradient = g(rhs_gradient)

    if isinstance(lhs_gradient, g.lattice) and isinstance(rhs_gradient, g.expr):
        rhs_otype = rhs_gradient.lattice().otype
        lhs_otype = lhs_gradient.otype
        if lhs_otype.__name__ != rhs_otype.__name__:
            if rhs_otype.spintrace[2] is not None:
                if lhs_otype.__name__ == rhs_otype.spintrace[2]().__name__:
                    rhs_gradient = g(g.spin_trace(rhs_gradient))
                    rhs_otype = rhs_gradient.otype
            if rhs_otype.colortrace[2] is not None:
                if lhs_otype.__name__ == rhs_otype.colortrace[2]().__name__:
                    rhs_gradient = g(g.color_trace(rhs_gradient))
                    rhs_otype = rhs_gradient.otype

    if setter is not None:
        setter(lhs.gradient, lhs_gradient + rhs_gradient)
    else:
        lhs.gradient += rhs_gradient
