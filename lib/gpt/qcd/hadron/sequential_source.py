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

class nucleon3pt_seq_src():

    def p2p_dbard(src, Pol_i, Spin_M, t_sink):

        tmp_seq_src = g.lattice(src)
        q1_tmp = g.eval(Pol_i * src * Spin_M)   # e.g.  Pol_i * D * Cg5
        q2_tmp = g.eval(Spin_M * src)           # e.g.  Cg5 * D
        tmp_seq_src = -qC.quarkContract14(q1_tmp, q2_tmp)

        q1_tmp = g.eval(q2_tmp * Spin_M)       #  e.g.  Cg5 * D * Cg5
        q2_tmp = g.eval(src * Pol_i)           #  e.g.  D * Pol_i
        tmp_seq_src -= g.spin_transpose(qC.quarkContract12(q2_tmp, q1_tmp))

        tmp_seq_src = g.eval(g.gamma[5] * g.adj(tmp_seq_src) * g.gamma[5])

        seq_src = g.lattice(src)
        seq_src[:] = 0
        seq_src[:, :, :, t_sink] = tmp_seq_src[:, :, :, t_sink]
        return seq_src

    def p2p_ubaru(src, Pol_i, Spin_M, t_sink):

        I = g.ot_matrix_spin_color(4, 3).identity()
        tmp_seq_src = g.lattice(src)
        q1_tmp = g.eval(src * Spin_M)           # e.g.  D * Cg5
        q2_tmp = g.eval(Spin_M * src)           # e.g.  Cg5 * D
        di_quark = qC.quarkContract24(q1_tmp, q2_tmp)

        tmp_seq_src = g.eval(Pol_i * di_quark)
        tmp_seq_src += g.eval(g.spin_trace(di_quark) * I * Pol_i)

        q1_tmp = g.eval(q2_tmp * Spin_M)       #  e.g.  Cg5 * D * Cg5
        q2_tmp = g.eval(src * Pol_i)           #  e.g.  D * Pol_i
        tmp_seq_src -= qC.quarkContract13(q1_tmp, q2_tmp)
        tmp_seq_src -= g.spin_transpose(qC.quarkContract12(q2_tmp, q1_tmp))

        tmp_seq_src @= g.eval(g.gamma[5] * g.adj(tmp_seq_src) * g.gamma[5])

        seq_src = g.lattice(src)
        seq_src[:] = 0
        seq_src[:, :, :, t_sink] = tmp_seq_src[:, :, :, t_sink]
        return seq_src


    def p2n_dbaru(src, Pol_i, Spin_M, t_sink):

        I = g.ot_matrix_spin_color(4, 3).identity()
        tmp_seq_src = g.lattice(src)
        q1_tmp = g.eval(src * Spin_M)           # e.g.  D * Cg5
        q2_tmp = g.eval(Spin_M * src)           # e.g.  Cg5 * D
        di_quark = qC.quarkContract24(q1_tmp, q2_tmp)

        tmp_seq_src = qC.quarkContract14(g.eval(Pol_i * q1_tmp), q2_tmp)

        tmp_seq_src += g.eval(Pol_i * di_quark)
        tmp_seq_src += g.eval(g.spin_trace(di_quark) * I * Pol_i)

        q1_tmp = q2_tmp * Spin_M                # e.g. (Cg5 * D) * Cg5
        q2_tmp = src * Pol_i                    # e.g. D * Pol_i

        tmp_seq_src -= qC.quarkContract13(q1_tmp, q2_tmp)

        seq_src = g.lattice(src)
        seq_src[:] = 0
        seq_src[:, :, :, t_sink] = tmp_seq_src[:, :, :, t_sink]
        return seq_src

