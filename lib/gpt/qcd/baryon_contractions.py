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


class BaryonsContractions2prop():

    def _proton_2pt(self, src_1, src_2, Pol_i, Spin_M):
        I = g.ot_matrix_spin_color(4, 3).identity()
        di_quark = qC.quarkContract13(g.eval(src_1 * Spin_M), g.eval(Spin_M * src_2))
        return  g.trace(g.eval(Pol_i * g.color_trace(src_2 * I * g.spin_trace(di_quark))) +
                        g.eval(Pol_i * g.color_trace(src_2 * di_quark)))


    def _xi_2pt(self, src_1 , src_2, Pol_i, Spin_M):
        I = g.ot_matrix_spin_color(4, 3).identity()
        di_quark = qC.quarkContract13(g.eval(src_1 * Spin_M), g.eval(Spin_M * src_2))
        return  g.trace(g.eval(Pol_i * g.color_trace(src_1 * I * g.spin_trace(di_quark))) + \
                        g.eval(Pol_i * g.color_trace(src_1 * di_quark)))


    def _lambda_2pt(self, src_1 , src_2, Pol_i, Spin_M):
        I = g.ot_matrix_spin_color(4, 3).identity()
        di_quark = qC.quarkContract13(g.eval(src_2 * Spin_M), g.eval(Spin_M * src_2))
        b_prop = g.trace(g.eval(Pol_i * g.color_trace(src_1 * I * g.spin_trace(di_quark))) + \
                         g.eval(Pol_i * g.color_trace(src_1 * di_quark)))
        di_quark = qC.quarkContract13(g.eval(src_2 * Spin_M), g.eval(Spin_M * src_1))
        b_prop += g.trace(g.eval(Pol_i * g.color_trace(src_2 * di_quark)))
        return b_prop


    def _lambda_naive_2pt(self, src_1 , src_2, Pol_i, Spin_M):
        I = g.ot_matrix_spin_color(4, 3).identity()
        di_quark = qC.quarkContract13(g.eval(src_2 * Spin_M), g.eval(Spin_M * src_2))
        return  g.trace(g.eval(Pol_i * g.color_trace(src_1 * I * g.spin_trace(di_quark))))


    def _sigma_star_2pt(self, src_1 , src_2, Pol_i, Spin_M):
        I = g.ot_matrix_spin_color(4, 3).identity()
        di_quark = qC.quarkContract13(g.eval(src_1 * Spin_M), g.eval(Spin_M * src_2))
        b_prop = g.trace(g.eval(Pol_i * g.color_trace(src_2 * g.spin_trace(di_quark))) + \
                         g.eval(Pol_i * g.color_trace(src_2 * di_quark)))
        di_quark = qC.quarkContract13(g.eval(src_2 * Spin_M), g.eval(Spin_M * src_1))
        b_prop += g.trace(g.eval(Pol_i * g.color_trace(src_2 * di_quark)))
        di_quark = qC.quarkContract13(g.eval(src_2 * Spin_M), g.eval(Spin_M * src_2))
        b_prop += g.trace(g.eval(Pol_i * g.color_trace(src_1 * di_quark)))
        b_prop *= 2
        b_prop += g.trace(g.eval(Pol_i * g.color_trace(src_1 * g.spin_trace(di_quark))))
        return b_prop


class BaryonsContractions3prop():

    def _sigma_2pt(self, src_1 , src_2, src_3, Pol_i, Spin_M):
        I = g.ot_matrix_spin_color(4, 3).identity()
        di_quark = qC.quarkContract13(g.eval(src_3 * Spin_M), g.eval(Spin_M * src_2))
        return  g.trace(g.eval(Pol_i * g.color_trace(src_1 * I * g.spin_trace(di_quark)))) + \
                        g.trace(g.eval(Pol_i * g.color_trace(src_1 * di_quark)))


    def _sigma_star(self, prop_u , prop_s, prop_d, Pol_i, Spin_M):
        I = g.ot_matrix_spin_color(4, 3).identity()
        return  g.trace(g.eval(Pol_i * prop_s * I * g.spin_trace(qC.quarkContract13(g.eval(prop_u * Spin_M), g.eval(Spin_M * prop_d)))) + \
                        g.eval(prop_u * Pol_i * qC.quarkContract24(prop_s, g.eval(Spin_M * prop_d * Spin_M))) +\
                        g.eval(Pol_i * prop_s * Pol_i * qC.quarkContract13(prop_u, g.eval(Spin_M * prop_d))))


    def _sigma_star_2pt(self, prop_u , prop_s, prop_d, Pol_i, Spin_M):
        return   self._sigma_star(prop_u, prop_s, prop_d, Pol_i, Spin_M) + \
                 self._sigma_star(prop_d, prop_u, prop_s, Pol_i, Spin_M) + \
                 self._sigma_star(prop_s, prop_d, prop_u, Pol_i, Spin_M)


    def _sigma0_2pt(self, prop_s , prop_u, prop_d, Pol_i, Spin_M):
        return 1 / 2 * (self._sigma_2pt(prop_u, prop_d, prop_s, Pol_i, Spin_M) + \
                        self._sigma_2pt(prop_d, prop_u, prop_s, Pol_i, Spin_M))


    def _lambda8_2pt(self, prop_h, prop_u, prop_d, Pol_i, Spin_M):
        return  (2. * self._sigma_2pt(prop_h, prop_d , prop_u, Pol_i, Spin_M) + \
                 2. * self._sigma_2pt(prop_h, prop_u, prop_d, Pol_i, Spin_M) + \
                 2. * self._sigma_2pt(prop_d, prop_h, prop_u, Pol_i, Spin_M) + \
                 2. * self._sigma_2pt(prop_u, prop_h, prop_d, Pol_i, Spin_M) - \
                      self._sigma_2pt(prop_d, prop_u, prop_h, Pol_i, Spin_M) - \
                      self._sigma_2pt(prop_u, prop_d, prop_h, Pol_i, Spin_M)) / 6.


    def _lambda8_to_sigma0_2pt(self, prop_s, prop_u, prop_d, Pol_i, Spin_M):
        return  (2. * self._sigma_2pt(prop_s, prop_d, prop_u, Pol_i, Spin_M) - \
                 2. * self._sigma_2pt(prop_s, prop_u, prop_d, Pol_i, Spin_M) + \
                      self._sigma_2pt(prop_d, prop_u, prop_s, Pol_i, Spin_M) - \
                      self._sigma_2pt(prop_u, prop_d, prop_s, Pol_i, Spin_M)) / np.sqrt(12.)


    def _sigma0_to_lambda8_2pt(self, prop_s, prop_u, prop_d, Pol_i, Spin_M):
        return  (2 * self._sigma_2pt(prop_d, prop_s, prop_u, Pol_i, Spin_M) - \
                 2 * self._sigma_2pt(prop_u, prop_s, prop_d, Pol_i, Spin_M) - \
                     self._sigma_2pt(prop_d, prop_u, prop_s, Pol_i, Spin_M) + \
                     self._sigma_2pt(prop_u, prop_d, prop_s, Pol_i, Spin_M)) / np.sqrt(12.)


