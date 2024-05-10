#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


class infinitesimal:
    def __init__(self, value):
        if isinstance(value, str):
            value = {value: 1}
        self.value = value

    def __pow__(self, n):
        value = {}
        for v in self.value:
            value[v] = self.value[v] * n
        return infinitesimal(value)

    def __mul__(self, other):
        value = {}
        for v1 in self.value:
            value[v1] = self.value[v1]
        for v2 in other.value:
            if v2 in value:
                value[v2] += other.value[v2]
            else:
                value[v2] = other.value[v2]
        return infinitesimal(value)

    def __str__(self):
        r = ""
        for v in sorted(self.value):
            if r != "":
                r = r + "*"
            if self.value[v] == 1:
                r = r + v
            else:
                r = r + f"{v}**{self.value[v]}"
        return r

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __cmp__(self, other):
        return self.__str__().__cmp__(other.__str__())

    def symbols(self):
        return tuple(sorted(list(self.value.keys())))

    def behaves_as(self, other):
        for s in other.value:
            n0 = self.value[s] if s in self.value else 0
            n1 = other.value[s]
            if n0 < n1:
                return False
        return True
