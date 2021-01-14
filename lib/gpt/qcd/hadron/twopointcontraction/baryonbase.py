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


# TODO clean up and check
def contract_one_flavor_baryon(prop_1, prop_2, prop_3, pol_matrix, source_spin_matrix, sink_spin_matrix):
    di_quark = quark_contract_13(gpt.eval(prop_3 * source_spin_matrix), gpt.eval(sink_spin_matrix * prop_2))
    return gpt.trace(
        gpt.eval(pol_matrix * gpt.color_trace(prop_1 * gpt.spin_trace(di_quark))) +
        gpt.eval(pol_matrix * gpt.color_trace(prop_1 * di_quark))
    )


# TODO clean up and check
def contract_two_flavor_baryon(prop_1, prop_2, prop_3, pol_matrix, source_spin_matrix, sink_spin_matrix):
    di_quark = quark_contract_13(gpt.eval(prop_3 * source_spin_matrix), gpt.eval(sink_spin_matrix * prop_2))
    return gpt.trace(
        gpt.eval(pol_matrix * gpt.color_trace(prop_1 * gpt.spin_trace(di_quark))) +
        gpt.eval(pol_matrix * gpt.color_trace(prop_1 * di_quark))
    )


# TODO clean up and check
def contract_three_flavor_baryon(prop_1, prop_2, prop_3, pol_matrix, source_spin_matrix, sink_spin_matrix):
    di_quark = quark_contract_13(gpt.eval(prop_1 * source_spin_matrix), gpt.eval(sink_spin_matrix * prop_2))
    return gpt.trace(gpt.eval(pol_matrix * gpt.color_trace(prop_3 * gpt.spin_trace(di_quark))))
