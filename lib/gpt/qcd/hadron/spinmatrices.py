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


def positive_parity_unpolarized():
    return 0.5 * (gpt.gamma["I"] + gpt.gamma["T"])


def positive_parity_xpolarized():
    return positive_parity_unpolarized() * -1j * gpt.gamma["X"] * gpt.gamma[5]


def positive_parity_ypolarized():
    return positive_parity_unpolarized() * -1j * gpt.gamma["Y"] * gpt.gamma[5]


def positive_parity_zpolarized():
    return positive_parity_unpolarized() * -1j * gpt.gamma["Z"] * gpt.gamma[5]


def negative_parity_unpolarized():
    return 0.5 * (gpt.gamma["I"] - gpt.gamma["T"])


def negative_parity_xpolarized():
    return negative_parity_unpolarized() * -1j * gpt.gamma["X"] * gpt.gamma[5]


def negative_parity_ypolarized():
    return negative_parity_unpolarized() * -1j * gpt.gamma["Y"] * gpt.gamma[5]


def negative_parity_zpolarized():
    return negative_parity_unpolarized() * -1j * gpt.gamma["Z"] * gpt.gamma[5]


def charge_conjugation():
    return gpt.gamma["Y"] * gpt.gamma["T"]


def gamma_minus():
    return 0.5 * (gpt.gamma["Y"] + 1j * gpt.gamma["X"])
