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
import gpt
from gpt.params import params_convention


@params_convention(parity=gpt.odd)
def eo1(params):
    parity = params["parity"]

    def instantiate(op):
        return gpt.algorithms.preconditioner.schur_complement_one(
            op, lambda op: op.even_odd_sites_decomposed(parity)
        )

    return instantiate


@params_convention(parity=gpt.odd)
def eo1_ne(params):
    parity = params["parity"]

    def instantiate(op):
        return gpt.algorithms.preconditioner.normal_equation(
            gpt.algorithms.preconditioner.schur_complement_one(
                op, lambda op: op.even_odd_sites_decomposed(parity)
            )
        )

    return instantiate


@params_convention(parity=gpt.odd)
def eo2(params):
    parity = params["parity"]

    def instantiate(op):
        return gpt.algorithms.preconditioner.schur_complement_two(
            op, lambda op: op.even_odd_sites_decomposed(parity)
        )

    return instantiate


@params_convention(parity=gpt.odd)
def eo2_ne(params):
    parity = params["parity"]

    def instantiate(op):
        return gpt.algorithms.preconditioner.normal_equation(
            gpt.algorithms.preconditioner.schur_complement_two(
                op, lambda op: op.even_odd_sites_decomposed(parity)
            )
        )

    return instantiate


@params_convention(parity=gpt.odd)
def eo2_kappa_ne(params):
    parity = params["parity"]

    def instantiate(op):
        return gpt.algorithms.preconditioner.normal_equation(
            gpt.algorithms.preconditioner.similarity_transformation(
                gpt.algorithms.preconditioner.schur_complement_two(
                    op, lambda op: op.even_odd_sites_decomposed(parity)
                ),
                op.kappa(),
            )
        )

    return instantiate
