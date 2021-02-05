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
from numpy import sqrt


def baryon_octet_base_contraction(prop, diquark, pol_matrix):
    return gpt.trace(
        pol_matrix * gpt.color_trace(prop * gpt.spin_trace(diquark)) +
        pol_matrix * gpt.color_trace(prop * diquark)
    )


def contract_proton(prop_up, prop_down, spin_matrix, pol_matrix, diquark=None):
    if diquark is None:
        diquark = quark_contract_13(
            gpt.eval(prop_down * spin_matrix), gpt.eval(spin_matrix * prop_up)
        )
    return baryon_octet_base_contraction(prop_up, diquark, pol_matrix)


def contract_neutron(prop_up, prop_down, spin_matrix, pol_matrix, diquark=None):
    if diquark is None:
        diquark = quark_contract_13(
            gpt.eval(prop_up * spin_matrix), gpt.eval(spin_matrix * prop_down)
        )
    return baryon_octet_base_contraction(prop_down, diquark, pol_matrix)


def contract_xi_zero(prop_up, prop_strange, spin_matrix, pol_matrix, diquark=None):
    if diquark is None:
        diquark = quark_contract_13(
            gpt.eval(prop_up * spin_matrix), gpt.eval(spin_matrix * prop_strange)
        )
    return baryon_octet_base_contraction(prop_strange, diquark, pol_matrix)


def contract_sigma_plus(prop_up, prop_strange, spin_matrix, pol_matrix, diquark=None):
    if diquark is None:
        diquark = quark_contract_13(
            gpt.eval(prop_strange * spin_matrix), gpt.eval(spin_matrix * prop_up)
        )
    return baryon_octet_base_contraction(prop_up, diquark, pol_matrix)


def contract_lambda(prop_up, prop_down, prop_strange, spin_matrix, pol_matrix, diquarks=None):
    if diquarks is None:
        diquarks = []
        prop_dict = {"up": prop_up, "down": prop_down, "strange": prop_strange}
        for diquark_flavors in [
            "up_down", "down_up",
            "up_strange", "down_strange",
            "strange_up", "strange_down"
        ]:
            flav1, flav2 = diquark_flavors.split("_")
            diquarks.append(quark_contract_13(
                gpt.eval(prop_dict[flav1] * spin_matrix), gpt.eval(spin_matrix * prop_dict[flav2])
            ))
    return (
        2 * baryon_octet_base_contraction(prop_strange, diquarks[0], pol_matrix) +
        2 * baryon_octet_base_contraction(prop_strange, diquarks[1], pol_matrix) +
        2 * baryon_octet_base_contraction(prop_down, diquarks[2], pol_matrix) +
        2 * baryon_octet_base_contraction(prop_up, diquarks[3], pol_matrix) -
        baryon_octet_base_contraction(prop_down, diquarks[4], pol_matrix) -
        baryon_octet_base_contraction(prop_up, diquarks[5], pol_matrix)
    ) / 6.


def contract_lambda_naive(prop_up, prop_down, prop_strange, spin_matrix, pol_matrix, diquark=None):
    if diquark is None:
        diquark = quark_contract_13(
            gpt.eval(prop_up * spin_matrix), gpt.eval(spin_matrix * prop_down)
        )
    return gpt.trace(pol_matrix * gpt.color_trace(prop_strange * gpt.spin_trace(diquark)))


def contract_lambda_to_sigma_zero(prop_up, prop_down, prop_strange, spin_matrix, pol_matrix, diquarks=None):
    if diquarks is None:
        diquarks = []
        prop_dict = {"up": prop_up, "down": prop_down, "strange": prop_strange}
        for diquark_flavors in [
            "up_down", "down_up",
            "strange_up", "strange_down"
        ]:
            flav1, flav2 = diquark_flavors.split("_")
            diquarks.append(quark_contract_13(
                gpt.eval(prop_dict[flav1] * spin_matrix), gpt.eval(spin_matrix * prop_dict[flav2])
            ))
    return (
        2 * baryon_octet_base_contraction(prop_strange, diquarks[0], pol_matrix) -
        2 * baryon_octet_base_contraction(prop_strange, diquarks[1], pol_matrix) +
        baryon_octet_base_contraction(prop_down, diquarks[2], pol_matrix) -
        baryon_octet_base_contraction(prop_up, diquarks[3], pol_matrix)
    ) / sqrt(12)

