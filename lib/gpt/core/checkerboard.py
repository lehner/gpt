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
import cgpt
import gpt


class even:
    # per convention sum(coordinates) % 2 == tag
    tag = 0

    def inv():
        return gpt.odd


class odd:
    tag = 1

    def inv():
        return gpt.even


class none:
    tag = None

    def inv():
        return gpt.none


def str_to_cb(s):
    if s == "even":
        return even
    elif s == "odd":
        return odd
    elif s == "none":
        return none
    else:
        assert 0


def pick_checkerboard(cb, dst, src):
    assert len(src.v_obj) == len(dst.v_obj)
    for i in src.otype.v_idx:
        cgpt.lattice_pick_checkerboard(cb.tag, src.v_obj[i], dst.v_obj[i])


def set_checkerboard(dst, src):
    assert len(src.v_obj) == len(dst.v_obj)
    for i in src.otype.v_idx:
        cgpt.lattice_set_checkerboard(src.v_obj[i], dst.v_obj[i])
