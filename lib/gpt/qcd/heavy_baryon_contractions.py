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

import gpt as g
from gpt.qcd import quarkContract as qC
import numpy as np

#
# Add reference and comments
#
# Generalize functions?
#

class HeavyBaryonsContractions2prop():

    #
    # O_mu = eps^{abc} * (q_1^a * (C * Gamma_{\mu}) * q_1^b) q_2^c
    #
    #
    def _sigmac_munu_2pt(self, src_1, src_2, Spin_M, Pol_mu, Pol_nu):
        I = g.ot_matrix_spin_color(4, 3).identity()
        di_quark = qC.quarkContract13(g.eval(src_1 * Pol_nu), g.eval(Pol_mu * src_1))
        heavy_quark_corr = g.trace(g.eval(Spin_M * I * g.color_trace(src_2 * I * g.spin_trace(di_quark))))
        di_quark = qC.quarkContract13(src_1, g.eval(Pol_mu * src_1 * Pol_nu))
        heavy_quark_corr = g.trace(src_1)
        heavy_quark_corr += g.trace(Spin_M * I * g.color_trace(src_2 * g.spin_trace(di_quark)))
        return  heavy_quark_corr


    #
    # O_A = eps^{abc} (q_2^a * Gamma2 * q_1^b) * Gamma1 * q_1^c
    # O_B = eps^{abc} (q_2^a * Gamma4 * q_1^b) * Gamma3 * q_1^c
    #
    # Suitable for N-like particles (P, Sigma^{+/-}, Xi^-, Xi^0, Omega_c^0, Sigma_c^{++}, ...)
    #

    def _sigmac_GT_2pt(self, src_1, src_2, Spin_M, G2, G4):
        I = g.ot_matrix_spin_color(4, 3).identity()
        di_quark = qC.quarkContract13(g.eval(src_2 * G4), g.eval(G2 * src_1))
        heavy_quark_corr = g.trace(Spin_M * I * g.color_trace(src_1 * di_quark))
        heavy_quark_corr += g.trace(Spin_M * I * g.color_trace(src_2 * g.spin_trace(di_quark)))
        return  heavy_quark_corr



class HeavyBaryonsContractions3prop():

    def _sigmac_munu_2pt(self, src_1 , src_2, src_3, Pol_i, Gmu, Gnu):
        I = g.ot_matrix_spin_color(4, 3).identity()
        di_quark = qC.quarkContract13(g.eval(src_1 * Gnu), g.eval(Gmu * src_2))
        return  g.trace(g.eval(Pol_i * g.color_trace(src_3 * I * g.spin_trace(di_quark))))


    def _sigmac_GT_2pt(self, src_1 , src_2, src_3, Pol_i, G2, G4):
        I = g.ot_matrix_spin_color(4, 3).identity()
        di_quark = qC.quarkContract13(g.eval(src_3 * G4), g.eval(G2 * src_1))
        heavy_quark_corr = g.trace(g.eval(Pol_i * g.color_trace(src_2 * di_quark)))
        heavy_quark_corr += g.trace(Pol_i * g.color_trace(src_2 * g.spin_trace(di_quark)))
        return heavy_quark_corr


    def _sigmac0_GT_2pt(self, src_1, src_2, src_3, Pol_i, G2, G4):
        heavy_quark_corr =  1/2 * (self._sigmac_GT_2pt(src_1, src_2, src_3, Pol_i, G2, G4) +
                                   self._sigmac_GT_2pt(src_2, src_1, src_3, Pol_i, G2, G4))
        return heavy_quark_corr


    def _lambdac_2pt(self, src_1, src_2, src_3, Pol_i, Spin_M):
        di_quark = qC.quarkContract13(g.eval(src_1 * Spin_M), g.eval(Spin_M * src_2))
        return g.trace(Pol_i * g.color_trace(src_3 * g.spin_trace(di_quark)))


    def _lambdac_2pt_v2(self, src_1, src_2, src_3, Pol_i, G2, G4):
        di_quark = qC.quarkContract13(g.eval(src_1 * G4), g.eval(G2 * src_2))
        return g.trace(Pol_i * g.color_trace(src_3 * g.spin_trace(di_quark)))


    def _lambda8_2pt(self, prop_h, prop_u, prop_d, Pol_i, G2, G4):
        return  (2. * self._sigmac_GT_2pt(prop_h, prop_d , prop_u, Pol_i, G2, G4) + \
                 2. * self._sigmac_GT_2pt(prop_h, prop_u, prop_d, Pol_i, G2, G4) + \
                 2. * self._sigmac_GT_2pt(prop_d, prop_h, prop_u, Pol_i, G2, G4) + \
                 2. * self._sigmac_GT_2pt(prop_u, prop_h, prop_d, Pol_i, G2, G4) - \
                      self._sigmac_GT_2pt(prop_d, prop_u, prop_h, Pol_i, G2, G4) - \
                      self._sigmac_GT_2pt(prop_u, prop_d, prop_h, Pol_i, G2, G4)) / 6.

