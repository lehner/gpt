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
from gpt.qcd.hadron.twopointcontraction.baryonoctet import \
  baryon_octet_base_contraction, contract_lambda, contract_lambda_naive, contract_lambda_to_sigma_zero
from gpt.qcd.hadron.quarkcontract import quark_contract_13


def contract_charmed_lambda(prop_up, prop_down, prop_charm, spin_matrix, pol_matrix, diquarks=None):
    return contract_lambda(prop_up, prop_down, prop_charm, spin_matrix, pol_matrix, diquarks=diquarks)


def contract_charmed_lambda_naive(prop_up, prop_down, prop_charm, spin_matrix, pol_matrix, diquark=None):
    return contract_lambda_naive(prop_up, prop_down, prop_charm, spin_matrix, pol_matrix, diquark)


def contract_charmed_sigma_zero(prop_down, prop_charm, spin_matrix, pol_matrix, diquark=None):
    if diquark is None:
        diquark = quark_contract_13(
            gpt.eval(prop_charm * spin_matrix), gpt.eval(spin_matrix * prop_down)
        )
    return baryon_octet_base_contraction(prop_down, diquark, pol_matrix)


def contract_charmed_xi_zero(prop_down, prop_strange, prop_charm, spin_matrix, pol_matrix, diquarks=None):
    return contract_lambda(prop_down, prop_strange, prop_charm, spin_matrix, pol_matrix, diquarks=diquarks)


def contract_charmed_xi_prime_zero(prop_down, prop_strange, prop_charm, spin_matrix, pol_matrix, diquarks=None):
    if diquarks is None:
        diquarks = [
            quark_contract_13(
                gpt.eval(prop_charm * spin_matrix), gpt.eval(spin_matrix * prop_strange)
            ),
            quark_contract_13(
                gpt.eval(prop_charm * spin_matrix), gpt.eval(spin_matrix * prop_down)
            )
        ]
    return 0.5 * (
        gpt.eval(baryon_octet_base_contraction(prop_down, diquarks[0], pol_matrix)) +
        gpt.eval(baryon_octet_base_contraction(prop_strange, diquarks[1], pol_matrix))
    )


def contract_double_charmed_xi_plus(prop_down, prop_charm, spin_matrix, pol_matrix, diquark=None):
    if diquark is None:
        diquark = quark_contract_13(
            gpt.eval(prop_down * spin_matrix), gpt.eval(spin_matrix * prop_charm)
        )
    return baryon_octet_base_contraction(prop_charm, diquark, pol_matrix)


def contract_charmed_omega(prop_strange, prop_charm, spin_matrix, pol_matrix, diquark=None):
    if diquark is None:
        diquark = quark_contract_13(
            gpt.eval(prop_charm * spin_matrix), gpt.eval(spin_matrix * prop_strange)
        )
    return baryon_octet_base_contraction(prop_strange, diquark, pol_matrix)


def contract_double_charmed_omega(prop_strange, prop_charm, spin_matrix, pol_matrix, diquark=None):
    if diquark is None:
        diquark = quark_contract_13(
            gpt.eval(prop_strange * spin_matrix), gpt.eval(spin_matrix * prop_charm)
        )
    return baryon_octet_base_contraction(prop_charm, diquark, pol_matrix)


def charmed_xi_to_xi_prime(prop_down, prop_strange, prop_charm, spin_matrix, pol_matrix, diquarks=None):
    return contract_lambda_to_sigma_zero(prop_down, prop_strange, prop_charm, spin_matrix, pol_matrix, diquarks=diquarks)

