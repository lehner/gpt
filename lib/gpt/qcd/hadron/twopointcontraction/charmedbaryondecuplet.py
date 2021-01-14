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
from gpt.qcd.hadron.twopointcontraction.baryonoctet import baryon_base_contraction
from gpt.qcd.hadron.twopointcontraction.baryondecuplet import contract_xi_zero_star
from gpt.qcd.hadron.spinmatrices import charge_conjugation, gamma_minus
from gpt.qcd.hadron.quarkcontract import quark_contract_13, quark_contract_24


def contract_charmed_sigma_star_zero(prop_down, prop_charm, pol_matrix):
    return contract_xi_zero_star(prop_charm, prop_down, pol_matrix)


def chroma_sigma_star(prop_1, prop_2, prop_3, pol_matrix, cgm):
    diquark = quark_contract_13(gpt.eval(prop_1 * cgm), gpt.eval(cgm * prop_3))
    contraction = gpt.trace(gpt.eval(pol_matrix * prop_2 * gpt.spin_trace(diquark)))

    diquark @= quark_contract_24(prop_2, gpt.eval(cgm * prop_3 * cgm))
    contraction += gpt.trace(gpt.eval(prop_1 * pol_matrix * diquark))

    diquark @= quark_contract_13(prop_1, gpt.eval(cgm * prop_3))
    contraction += gpt.trace(gpt.eval(pol_matrix * prop_2 * cgm * diquark))
    return gpt.eval(contraction)


def contract_charmed_xi_zero_star(prop_down, prop_strange, prop_charm, pol_matrix):
    cgminus = charge_conjugation() * gamma_minus()
    return gpt.eval(
        chroma_sigma_star(prop_down, prop_strange, prop_charm, pol_matrix, cgminus) +
        chroma_sigma_star(prop_charm, prop_down, prop_strange, pol_matrix, cgminus) +
        chroma_sigma_star(prop_strange, prop_charm, prop_down, pol_matrix, cgminus)
    )


def contract_double_charmed_xi_plus_star(prop_down, prop_charm, pol_matrix):
    return contract_xi_zero_star(prop_down, prop_charm, pol_matrix)


def contract_charmed_omega_star(prop_strange, prop_charm, pol_matrix):
    return contract_xi_zero_star(prop_charm, prop_strange, pol_matrix)


def contract_double_charmed_omega_star(prop_strange, prop_charm, pol_matrix):
    return contract_xi_zero_star(prop_strange, prop_charm, pol_matrix)


def contract_triple_charmed_omega(prop_charm, pol_matrix):
    # TODO normalization?
    return contract_xi_zero_star(prop_charm, prop_charm, pol_matrix)
