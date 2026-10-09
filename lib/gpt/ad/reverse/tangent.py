#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
# Tangent rules (jvp) shared by the primitives (see primitive.py; used by
# g.ad.reverse.jacobian): a rule maps (z, children, tangents, **static) to
# the k tangents of z, given per child None (a constant) or its k tangents.
#
import operator


def count(tangents):
    # the number of tangents (of the children that have tangents)
    return len(next(t for t in tangents if t is not None))


def total(terms, add=operator.add):
    # the sum of the terms that are not None (None: no term)
    result = None
    for t in terms:
        if t is not None:
            result = t if result is None else add(result, t)
    return result


def constant(z, children, tangents, **static):
    # a value that does not depend on its arguments: no tangent
    return [None] * count(tangents)


def linear(op):
    # z = op(x), linear: dz = op(dx)
    return lambda z, children, tangents, **static: [op(t, **static) for t in tangents[0]]


def bilinear(op, add=operator.add):
    # z = op(x, y), bilinear: dz = op(dx, y) + op(x, dy)
    def rule(z, children, tangents, **static):
        (x, y), (tx, ty) = children, tangents
        return [
            total(
                [
                    None if tx is None else op(tx[j], y, **static),
                    None if ty is None else op(x, ty[j], **static),
                ],
                add,
            )
            for j in range(count(tangents))
        ]

    return rule
