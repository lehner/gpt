#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020-22  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


def sqrt(A):
    # Denman and Beavers 1976
    Mk = A
    Xk = gpt.identity(A)
    norm = gpt.norm2(Xk)
    for k in range(100):
        Xkp1 = gpt(0.5 * Xk + 0.5 * gpt.matrix.inv(Mk))
        Mkp1 = gpt(0.5 * Mk + 0.5 * gpt.matrix.inv(Xk))

        eps = (gpt.norm2(Mk - Mkp1) / norm) ** 0.5
        Xk = Xkp1
        Mk = Mkp1

        if eps < 1e3 * A.grid.precision.eps:
            # gpt.message(f"Converged after {k} iterations")
            # one final Newton iteration (including the original A to avoid error accumulation)
            Mk = gpt(0.5 * Mk + 0.5 * gpt.matrix.inv(Mk) * A)

            # compute error
            # gpt.message("err",gpt.norm2(Mk * Mk - A) / gpt.norm2(A))
            return Mk

    gpt.message("Warning: sqrt not converged")
    return Mk
