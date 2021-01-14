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
from gpt.qcd.hadron.spinmatrices import charge_conjugation
from numpy import sqrt


def baryon_base_contraction(prop_1, prop_2, prop_3, pol_matrix, spin_matrix):
    di_quark = quark_contract_13(gpt.eval(prop_3 * spin_matrix), gpt.eval(spin_matrix * prop_2))
    return gpt.trace(
        gpt.eval(pol_matrix * gpt.color_trace(prop_1 * gpt.spin_trace(di_quark))) +
        gpt.eval(pol_matrix * gpt.color_trace(prop_1 * di_quark))
    )


def contract_proton(prop_up, prop_down, pol_matrix):
    cg5 = charge_conjugation() * gpt.gamma[5]
    return baryon_base_contraction(prop_up, prop_up, prop_down, pol_matrix, cg5)


def contract_neutron(prop_up, prop_down, pol_matrix):
    cg5 = charge_conjugation() * gpt.gamma[5]
    return baryon_base_contraction(prop_down, prop_down, prop_up, pol_matrix, cg5)


def contract_xi_zero(prop_up, prop_strange, pol_matrix):
    cg5 = charge_conjugation() * gpt.gamma[5]
    return baryon_base_contraction(prop_strange, prop_strange, prop_up, pol_matrix, cg5)


def contract_sigma_plus(prop_up, prop_strange, pol_matrix):
    cg5 = charge_conjugation() * gpt.gamma[5]
    return baryon_base_contraction(prop_up, prop_up, prop_strange, pol_matrix, cg5)


def contract_lambda(prop_up, prop_down, prop_strange, pol_matrix):
    cg5 = charge_conjugation() * gpt.gamma[5]
    return (
        2 * baryon_base_contraction(prop_strange, prop_down, prop_up, pol_matrix, cg5) +
        2 * baryon_base_contraction(prop_strange, prop_up, prop_down, pol_matrix, cg5) +
        2 * baryon_base_contraction(prop_down, prop_strange, prop_up, pol_matrix, cg5) +
        2 * baryon_base_contraction(prop_up, prop_strange, prop_down, pol_matrix, cg5) -
        baryon_base_contraction(prop_down, prop_up, prop_strange, pol_matrix, cg5) -
        baryon_base_contraction(prop_up, prop_down, prop_strange, pol_matrix, cg5)
    ) / 6.


def chroma_contract_lambda(prop_up, prop_down, prop_strange, pol_matrix):
    cg5 = charge_conjugation() * gpt.gamma[5]
    return (
        2 * baryon_base_contraction(prop_down, prop_strange, prop_up, pol_matrix, cg5) +
        2 * baryon_base_contraction(prop_down, prop_up, prop_strange, pol_matrix, cg5) +
        2 * baryon_base_contraction(prop_strange, prop_down, prop_up, pol_matrix, cg5) +
        2 * baryon_base_contraction(prop_up, prop_down, prop_strange, pol_matrix, cg5) -
        baryon_base_contraction(prop_strange, prop_up, prop_down, pol_matrix, cg5) -
        baryon_base_contraction(prop_up, prop_strange, prop_down, pol_matrix, cg5)
    ) / 6.


def contract_lambda_naive(prop_up, prop_down, prop_strange, pol_matrix):
    cg5 = charge_conjugation() * gpt.gamma[5]
    di_quark = quark_contract_13(gpt.eval(prop_up * cg5), gpt.eval(cg5 * prop_down))
    return gpt.trace(gpt.eval(pol_matrix * gpt.color_trace(prop_strange * gpt.spin_trace(di_quark))))


def contract_lambda_to_sigma_zero(prop_up, prop_down, prop_strange, pol_matrix):
    cg5 = charge_conjugation() * gpt.gamma[5]
    return (
        2 * baryon_base_contraction(prop_strange, prop_down, prop_up, pol_matrix, cg5) -
        2 * baryon_base_contraction(prop_strange, prop_up, prop_down, pol_matrix, cg5) +
        baryon_base_contraction(prop_down, prop_up, prop_strange, pol_matrix, cg5) -
        baryon_base_contraction(prop_up, prop_down, prop_strange, pol_matrix, cg5)
    ) / sqrt(12)
