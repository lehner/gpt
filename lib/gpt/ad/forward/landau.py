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


class landau:
    def __init__(self, *infinitesimals):
        self.infinitesimals = infinitesimals

    def accept(self, i):
        for j in self.infinitesimals:
            if i.behaves_as(j):
                return False
        return True

    def __add__(self, other):
        if self is other:
            return self
        infinitesimals = []
        for i in self.infinitesimals + other.infinitesimals:
            keep = True
            for n, j in enumerate(infinitesimals):
                if i.behaves_as(j):
                    keep = False
                elif j.behaves_as(i):
                    infinitesimals[n] = i
            if keep:
                infinitesimals.append(i)
        infinitesimals = list(set(infinitesimals))
        return landau(*infinitesimals)

    def __str__(self):
        a = []
        for i in self.infinitesimals:
            a.append(str(i))
        r = ",".join(sorted(a))
        return f"O({r})"

    def __eq__(self, other):
        return str(self) == str(other)
