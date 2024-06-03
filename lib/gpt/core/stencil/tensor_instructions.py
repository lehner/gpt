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
mov = 0
inc = 1
mov_neg = 2
dec = 3
mov_cc = 4
inc_cc = 5
mov_neg_cc = 6
dec_cc = 7
mul = 8
add = 9


def matrix_multiply(code, ndim, factor, idst, ifirst, isecond):
    for ia in range(ndim):
        for ib in range(ndim):
            dst = ia * ndim + ib
            for ic in range(ndim):
                aa = ia * ndim + ic
                bb = ic * ndim + ib
                mode = mov if ic == 0 else inc
                code.append((idst, dst, mode, 1.0, [(ifirst, 0, aa), (isecond, 0, bb)]))

            if factor != 1.0:
                code.append((idst, dst, mul, factor, [(idst, 0, dst)]))


def matrix_anti_hermitian(code, ndim, idst, isrc):
    for ia in range(ndim):
        for ib in range(ndim):
            dst = ia * ndim + ib
            dst_adj = ib * ndim + ia
            code.append((idst, dst, mov, 1.0, [(isrc, 0, dst)]))
            code.append((idst, dst, dec_cc, 1.0, [(isrc, 0, dst_adj)]))
            code.append((idst, dst, mul, 0.5, [(idst, 0, dst)]))


def matrix_trace(code, ndim, dst, factor, idst, isrc):
    for ia in range(ndim):
        mode = mov if ia == 0 else inc
        src = ia * ndim + ia
        code.append((idst, dst, mode, 1.0, [(isrc, 0, src)]))
    if factor != 1.0:
        code.append((idst, dst, mul, factor, [(isrc, 0, 0)]))


def matrix_trace_ab(code, ndim, dst, factor, idst, ifirst, isecond):
    mode = mov
    for ia in range(ndim):
        for ib in range(ndim):
            aa = ia * ndim + ib
            bb = ib * ndim + ia
            code.append((idst, dst, mode, 1.0, [(ifirst, 0, aa), (isecond, 0, bb)]))
            mode = inc
    if factor != 1.0:
        code.append((idst, dst, mul, factor, [(idst, 0, dst)]))


def matrix_diagonal_subtract(code, ndim, idst, isrc):
    for ia in range(ndim):
        src = ia * ndim + ia
        code.append((idst, src, dec, 1.0, [(isrc, 0, 0)]))
