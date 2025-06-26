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
import cgpt, gpt
import numpy as np

global_sum_algorithm = gpt.default.get("--global-sum-algorithm", "default")
fingerprint = gpt.default.get_int("--fingerprint", 0) > 0


def global_sum_grid(grid, x):
    if isinstance(x, gpt.tensor):
        cgpt.grid_globalsum(grid.obj, x.array)
        return x
    else:
        return cgpt.grid_globalsum(grid.obj, x)


verbose = True


def global_sum_reduce(x, reduce_function):
    global verbose
    if verbose:
        gpt.message(f"Using {global_sum_algorithm} global sums")
        verbose = False

    if isinstance(x, gpt.tensor):
        global_sum_reduce(x.array, reduce_function)
        return x

    if isinstance(x, np.ndarray):
        y = reduce_function(x, lambda a, b: a.__iadd__(b))
        np.copyto(x, y)
        return y
    elif isinstance(x, complex):
        x = np.array([x], dtype=np.complex128)
        global_sum_reduce(x, reduce_function)
        return complex(x[0])
    elif isinstance(x, float):
        x = np.array([x], dtype=np.float64)
        global_sum_reduce(x, reduce_function)
        return float(x[0])
    elif isinstance(x, int):
        x = np.array([x], dtype=np.int64)
        global_sum_reduce(x, reduce_function)
        return int(x[0])
    else:
        raise Exception(f"Unknown data type in global sum: {type(x)}")

    
def global_sum_binary_tree(grid, x):
    return global_sum_reduce(x, lambda y, functor: grid.reduce_binary_tree(y, functor))


def global_sum_dimensional_reduction(grid, x):
    return global_sum_reduce(x, lambda y, functor: grid.reduce_dimensional_reduction(y, functor))


global_sum_default = {"default": global_sum_grid, "binary-tree": global_sum_binary_tree, "dimensional-reduction": global_sum_dimensional_reduction}[
    global_sum_algorithm
]


if fingerprint:
    global_sum_fingerprint = global_sum_default

    def global_sum_default(grid, x):
        l = gpt.fingerprint.log()
        l("local", x)
        r = global_sum_fingerprint(grid, x)
        l("global", r)
        l()
        return r
