#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt as g
import numpy as np


def global_sum_quadruple(grid, data):
    # first promote
    if isinstance(data, complex):
        data = g.qcomplex(data)
    elif isinstance(data, float):
        data = g.qfloat(data)
    elif isinstance(data, np.ndarray):
        if data.dtype == np.float64:
            data = g.qfloat_array(data)
        elif data.dtype == np.complex128:
            data = g.qcomplex_array(data)
        else:
            raise NotImplementedError(
                f"Numpy array data type {data.dtype} not yet implemented in global_sum_quadruple"
            )
    else:
        raise NotImplementedError(
            f"Data type {type(data)} not yet implemented in global_sum_quadruple"
        )

    # then convert
    data_serial = np.ascontiguousarray(data.to_serial())

    def _red(a, b):
        a[:] = (data.from_serial(a) + data.from_serial(b)).to_serial()[:]

    # then reduce + convert back
    return data.from_serial(grid.reduce(data_serial, _red))
