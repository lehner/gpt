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


def defect(field):
    return field.otype.defect(field)


def cartesian(field):
    if isinstance(field, list):
        return [cartesian(f) for f in field]
    return g.lattice(field.grid, field.otype.cartesian()).checkerboard(
        field.checkerboard()
    )


def projected_convert(x, otype):
    return g.project(g.convert(x, otype), "defect")


def compose(left, right):
    as_list = isinstance(left, list)

    left = g.util.to_list(g(left))
    right = g.util.to_list(g(right))

    left_type = left[0].otype
    right_type = right[0].otype

    if left_type.__name__ == right_type.__name__:
        # if both are of same type, use common group compose and return
        dst = [left_type.compose(l, r) for l, r in zip(left, right)]

    else:
        # if they are not, see if either is a cartesian element of the other type
        left_type_cartesian = left_type.cartesian()
        right_type_cartesian = right_type.cartesian()

        if left_type.__name__ == right_type_cartesian.__name__:
            dst = [
                right_type.compose(projected_convert(l, right_type), r)
                for l, r in zip(left, right)
            ]

        elif left_type_cartesian.__name__ == right_type.__name__:
            dst = [
                left_type.compose(l, projected_convert(r, left_type))
                for l, r in zip(left, right)
            ]

        else:
            raise TypeError(
                f"{left_type.__name__} and {right_type.__name__} are not composable"
            )

    if as_list:
        return dst

    return dst[0]


def inner_product(left, right):
    if isinstance(left, list):
        return sum([inner_product(x, y) for x, y in zip(left, right)])
    # inner product over group's real vector space
    left_type = left.otype
    return left_type.inner_product(left, right)
