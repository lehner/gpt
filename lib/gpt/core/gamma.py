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
import gpt

class gamma_base:
    def __init__(self, gamma):
        self.gamma = gamma

    def __mul__(self, other):
        return gpt.expr(self).__mul__(other)

    def __rmul__(self, other):
        return gpt.expr(other).__mul__(self)

gamma = {
    0 : gamma_base(0),
    1 : gamma_base(1),
    2 : gamma_base(2),
    3 : gamma_base(3),
    5 : gamma_base(4),
    "X" : gamma_base(0),
    "Y" : gamma_base(1),
    "Z" : gamma_base(2),
    "T" : gamma_base(3),
}
