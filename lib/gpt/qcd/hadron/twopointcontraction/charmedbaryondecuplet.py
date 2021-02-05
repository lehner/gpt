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
from gpt.qcd.hadron.twopointcontraction.baryondecuplet import baryon_decuplet_base_contraction
from gpt.qcd.hadron.quarkcontract import quark_contract_13, quark_contract_24


def contract_charmed_sigma_star_zero(prop_down, prop_charm, spin_matrix, pol_matrix, diquarks=None):
    if diquarks is None:
        diquarks = [
            quark_contract_13(
                gpt.eval(prop_charm * spin_matrix), gpt.eval(spin_matrix * prop_down)
            ),
            quark_contract_13(
                gpt.eval(prop_down * spin_matrix), gpt.eval(spin_matrix * prop_charm)
            ),
            quark_contract_13(
                gpt.eval(prop_down * spin_matrix), gpt.eval(spin_matrix * prop_down)
            )
        ]
    return baryon_decuplet_base_contraction(prop_charm, prop_down, diquarks, pol_matrix)


def chroma_sigma_star(prop_1, prop_2, prop_3, spm, polm, diquark):
    #diquark = quark_contract_13(gpt.eval(prop_1 * spm), gpt.eval(spm * prop_3))
    contraction = gpt.trace(gpt.eval(polm * prop_2 * gpt.spin_trace(diquark)))
    diquark2 = quark_contract_24(prop_2, gpt.eval(spm * prop_3 * spm))
    contraction += gpt.trace(gpt.eval(prop_1 * polm * diquark2))
    diquark2 @= quark_contract_13(prop_1, gpt.eval(spm * prop_3))
    contraction += gpt.trace(gpt.eval(polm * prop_2 * spm * diquark2))
    return gpt.eval(contraction)


def contract_charmed_xi_zero_star(prop_down, prop_strange, prop_charm, spin_matrix, pol_matrix, diquarks=None):
    if diquarks is None:
        diquarks = [
            quark_contract_13(
                gpt.eval(prop_down * spin_matrix), gpt.eval(spin_matrix * prop_charm)
            ),
            quark_contract_13(
                gpt.eval(prop_charm * spin_matrix), gpt.eval(spin_matrix * prop_strange)
            ),
            quark_contract_13(
                gpt.eval(prop_strange * spin_matrix), gpt.eval(spin_matrix * prop_down)
            )
        ]
    return gpt.eval(
        chroma_sigma_star(prop_down, prop_strange, prop_charm, spin_matrix, pol_matrix, diquarks[0]) +
        chroma_sigma_star(prop_charm, prop_down, prop_strange, spin_matrix, pol_matrix, diquarks[1]) +
        chroma_sigma_star(prop_strange, prop_charm, prop_down, spin_matrix, pol_matrix, diquarks[2])
    )


def contract_double_charmed_xi_plus_star(prop_down, prop_charm, spin_matrix, pol_matrix, diquarks=None):
    if diquarks is None:
        diquarks = [
            quark_contract_13(
                gpt.eval(prop_down * spin_matrix), gpt.eval(spin_matrix * prop_charm)
            ),
            quark_contract_13(
                gpt.eval(prop_charm * spin_matrix), gpt.eval(spin_matrix * prop_down)
            ),
            quark_contract_13(
                gpt.eval(prop_charm * spin_matrix), gpt.eval(spin_matrix * prop_charm)
            )
        ]
    return baryon_decuplet_base_contraction(prop_down, prop_charm, diquarks, pol_matrix)


def contract_charmed_omega_star(prop_strange, prop_charm, spin_matrix, pol_matrix, diquarks=None):
    if diquarks is None:
        diquarks = [
            quark_contract_13(
                gpt.eval(prop_charm * spin_matrix), gpt.eval(spin_matrix * prop_strange)
            ),
            quark_contract_13(
                gpt.eval(prop_strange * spin_matrix), gpt.eval(spin_matrix * prop_charm)
            ),
            quark_contract_13(
                gpt.eval(prop_strange * spin_matrix), gpt.eval(spin_matrix * prop_strange)
            )
        ]
    return baryon_decuplet_base_contraction(prop_charm, prop_strange, diquarks, pol_matrix)


def contract_double_charmed_omega_star(prop_strange, prop_charm, spin_matrix, pol_matrix, diquarks=None):
    if diquarks is None:
        diquarks = [
            quark_contract_13(
                gpt.eval(prop_strange * spin_matrix), gpt.eval(spin_matrix * prop_charm)
            ),
            quark_contract_13(
                gpt.eval(prop_charm * spin_matrix), gpt.eval(spin_matrix * prop_strange)
            ),
            quark_contract_13(
                gpt.eval(prop_charm * spin_matrix), gpt.eval(spin_matrix * prop_charm)
            )
        ]
    return baryon_decuplet_base_contraction(prop_strange, prop_charm, diquarks, pol_matrix)


def contract_triple_charmed_omega(prop_charm, spin_matrix, pol_matrix, diquark=None):
    if diquark is None:
        diquark = quark_contract_13(
            gpt.eval(prop_charm * spin_matrix), gpt.eval(spin_matrix * prop_charm)
        )
    return baryon_decuplet_base_contraction(prop_charm, prop_charm, [diquark for _ in range(3)], pol_matrix)
