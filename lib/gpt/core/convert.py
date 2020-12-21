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
    if second in [gpt.single, gpt.double]:

        # if first is a list, distribute
        if type(first) == list:
            return [convert(x, second) for x in first]

        # if first is no list, evaluate
        src = gpt.eval(first)
        dst_grid = src.grid.converted(second)
        return convert(gpt.lattice(dst_grid, src.otype), src)

    elif isinstance(second, gpt.ot_base):

        # if first is a list, distribute
        if type(first) == list:
            return [convert(x, second) for x in first]

        # if first is no list, evaluate
        src = gpt.eval(first)
        if src.otype.__name__ == second.__name__:
            return src
        return convert(gpt.lattice(src.grid, second), src)

    elif type(first) == list:

        assert len(first) == len(second)
        for i in range(len(first)):
            convert(first[i], second[i])
        return first

    elif type(first) == gpt.lattice:

        # second may be expression
        second = gpt.eval(second)

        # if otypes differ, attempt otype conversion first
        if first.otype.__name__ != second.otype.__name__:
            assert first.otype.__name__ in second.otype.ctab
            tmp = gpt.lattice(first)
            second.otype.ctab[first.otype.__name__](tmp, second)
            second = tmp
            assert first.otype.__name__ == second.otype.__name__

        # convert precision if needed
        if first.grid == second.grid:
            gpt.copy(first, second)

        else:
            assert len(first.otype.v_idx) == len(second.otype.v_idx)
            for i in first.otype.v_idx:
                cgpt.convert(first.v_obj[i], second.v_obj[i])
            first.checkerboard(second.checkerboard())

        return first

    else:
        assert 0
