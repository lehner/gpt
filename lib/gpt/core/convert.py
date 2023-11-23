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
import cgpt, gpt, numpy, sys


def convert(first, second):
    # distribute if needed
    if isinstance(first, list):
        if isinstance(second, list):
            assert len(first) == len(second)
            for i in range(len(first)):
                convert(first[i], second[i])
            return first
        else:
            return [convert(x, second) for x in first]

    if isinstance(first, gpt.expr):
        first = gpt(first)

    return first.__class__.foundation.convert(first, second)
