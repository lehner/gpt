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
import gpt
import cgpt


def rank_inner_product(a, b, use_accelerator):
    a = [gpt.eval(x) for x in a]
    b = [gpt.eval(x) for x in b]
    otype = a[0].otype
    assert len(otype.v_idx) == len(b[0].otype.v_idx)
    return cgpt.lattice_rank_inner_product(a, b, use_accelerator)


def inner_product(a, b, use_accelerator):
    return a[0].grid.globalsum(rank_inner_product(a, b, use_accelerator))
