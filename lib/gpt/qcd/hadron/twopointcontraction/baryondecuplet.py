#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Lorenzo Barca    (lorenzo1.barca@ur.de)
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
from gpt.qcd.hadron.quarkcontract import quark_contract_13


def baryon_decuplet_base_contraction(prop_1, prop_2, diquarks, pol_matrix):
    assert isinstance(diquarks, list)

    contraction = gpt.trace(
        gpt.eval(pol_matrix * gpt.color_trace(prop_2 * gpt.spin_trace(diquarks[0]))) +
        gpt.eval(pol_matrix * gpt.color_trace(prop_2 * diquarks[0]))
    )
    contraction += gpt.eval(gpt.trace(pol_matrix * gpt.color_trace(prop_2 * diquarks[1])))
    contraction += gpt.eval(gpt.trace(pol_matrix * gpt.color_trace(prop_1 * diquarks[2])))
    contraction *= 2
    contraction += gpt.eval(gpt.trace(pol_matrix * gpt.color_trace(prop_1 * gpt.spin_trace(diquarks[2]))))
    return contraction


def contract_delta_plus(prop_up, prop_down, spin_matrix, pol_matrix, diquarks=None):
    if diquarks is None:
        diquarks = [
            quark_contract_13(
                gpt.eval(prop_up * spin_matrix), gpt.eval(spin_matrix * prop_down)
            ),
            quark_contract_13(
                gpt.eval(prop_down * spin_matrix), gpt.eval(spin_matrix * prop_up)
            ),
            quark_contract_13(
                gpt.eval(prop_down * spin_matrix), gpt.eval(spin_matrix * prop_down)
            )
        ]
    return baryon_decuplet_base_contraction(prop_up, prop_down, diquarks, pol_matrix)


def contract_xi_zero_star(prop_up, prop_strange, spin_matrix, pol_matrix, diquarks=None):
    if diquarks is None:
        diquarks = [
            quark_contract_13(
                gpt.eval(prop_up * spin_matrix), gpt.eval(spin_matrix * prop_strange)
            ),
            quark_contract_13(
                gpt.eval(prop_strange * spin_matrix), gpt.eval(spin_matrix * prop_up)
            ),
            quark_contract_13(
                gpt.eval(prop_strange * spin_matrix), gpt.eval(spin_matrix * prop_strange)
            )
        ]
    return baryon_decuplet_base_contraction(prop_up, prop_strange, diquarks, pol_matrix)


def contract_sigma_plus_star(prop_up, prop_strange, spin_matrix, pol_matrix, diquarks=None):
    if diquarks is None:
        diquarks = [
            quark_contract_13(
                gpt.eval(prop_strange * spin_matrix), gpt.eval(spin_matrix * prop_up)
            ),
            quark_contract_13(
                gpt.eval(prop_up * spin_matrix), gpt.eval(spin_matrix * prop_strange)
            ),
            quark_contract_13(
                gpt.eval(prop_up * spin_matrix), gpt.eval(spin_matrix * prop_up)
            )
        ]
    return baryon_decuplet_base_contraction(prop_strange, prop_up, diquarks, pol_matrix)


def contract_omega(prop_strange, spin_matrix, pol_matrix, diquark=None):
    if diquark is None:
        diquark = quark_contract_13(
            gpt.eval(prop_strange * spin_matrix), gpt.eval(spin_matrix * prop_strange)
        )
    return baryon_decuplet_base_contraction(prop_strange, prop_strange, [diquark for _ in range(3)], pol_matrix)

