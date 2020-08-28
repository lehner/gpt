#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
#    Authors:  Lorenzo Barca     2020
#              Christoph Lehner  2020
#
import gpt as g

class spin_matrix():

    def T_polx():
        return - 1 / 2 * 1j * (g.gamma["Y"] * g.gamma["Z"] + g.gamma["X"] * g.gamma[5])

    def T_poly():
        return 1 / 2 * 1j * (g.gamma["X"] * g.gamma["Z"] - g.gamma["Y"] * g.gamma[5])

    def T_polz():
        return - 1 / 2 * 1j * (g.gamma["X"] * g.gamma["Y"] + g.gamma["Z"] * g.gamma[5])

    def T_unpol():
        return 1 / 2 * (g.gamma["I"] + g.gamma["T"])

    def T_unpol_negpar():
        return  1 / 2 * (g.gamma["I"] - g.gamma["T"])

    def T_mixed():
        return 1 / 2 * (g.gamma["I"] + g.gamma["T"] - \
                        1j * (g.gamma["X"] * g.gamma["Y"] + g.gamma["Z"] * g.gamma[5]) )

    def T_mixed_negpar():
        return 1 / 2 * (g.gamma["I"] - g.gamma["T"] - \
                        1j * (g.gamma["X"] * g.gamma["Y"] - g.gamma["Z"] * g.gamma[5]) )

    def C():
        return g.gamma["Y"] * g.gamma["T"]

    def Cg5():
        return g.gamma["X"] * g.gamma["Z"]

    def Cg5g4():
        return g.gamma["Y"] * g.gamma[5]

    def Cgm():
        return 1 / 2 * g.gamma["Y"] * g.gamma["T"] * (g.gamma["Y"] + 1j * g.gamma["X"])

    def Cg5_NR():
        return spin_matrix.Cg5() * spin_matrix.T_unpol()

    def Cg5_NR_negpar():
        return spin_matrix.Cg5() * spin_matrix.T_unpol_negpar()

