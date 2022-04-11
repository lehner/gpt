#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2022
#
import gpt as g

# define symbols
alpha = (0, 4)
a = (1, 3)
b = (2, 3)

# define a tensor product basis
x = g.sparse_tensor.basis([alpha, a, b])
g.message(f"basis: {x}")

# define 4 parallel tensors on basis x
t = g.sparse_tensor.tensor(x, 4)

# check basis
assert len(x) == 3
assert x[0] == alpha
assert x[1] == a
assert x[2] == b

# set tensor elements (in parallel for all 4)
t[0, 1, 0] = [2.5, 1.0, 0.5, 2]
t[0, 1, 1] = [-2.5, 1.0, -0.5, 2]
t[2, 0, 0] = -1j

# test tensor contents
g.message(f"tensor: {t}")
assert t[:] == [
    {(0, 1, 1): (-2.5 + 0j), (2, 0, 0): (-0 - 1j), (0, 1, 0): (2.5 + 0j)},
    {(0, 1, 1): (1 + 0j), (2, 0, 0): (-0 - 1j), (0, 1, 0): (1 + 0j)},
    {(0, 1, 1): (-0.5 + 0j), (2, 0, 0): (-0 - 1j), (0, 1, 0): (0.5 + 0j)},
    {(0, 1, 1): (2 + 0j), (2, 0, 0): (-0 - 1j), (0, 1, 0): (2 + 0j)},
]

# test if missing element is properly returned
assert t[1, 2, 0] == [None, None, None, None]

# test if existing elements are properly returned
assert t[0, 1, 0] == [2.5, 1.0, 0.5, 2]

# create second tensor, a vector in alpha
t2 = g.sparse_tensor.tensor(g.sparse_tensor.basis([alpha]), 4)

t2[0] = 0.5
t2[2] = 2.0
t2[1] = 0.2

# multiply
v = t * t2

assert v[:] == [
    {(0, 1, 0): (1.25 + 0j), (2, 0, 0): -2j, (0, 1, 1): (-1.25 + 0j)},
    {(0, 1, 0): (0.5 + 0j), (2, 0, 0): -2j, (0, 1, 1): (0.5 + 0j)},
    {(0, 1, 0): (0.25 + 0j), (2, 0, 0): -2j, (0, 1, 1): (-0.25 + 0j)},
    {(0, 1, 0): (1 + 0j), (2, 0, 0): -2j, (0, 1, 1): (1 + 0j)},
]

assert len(v.basis) == 3
assert v.basis[0] == alpha
assert v.basis[1] == a
assert v.basis[2] == b

# test tensor contractions
cc = g.sparse_tensor.contract([v], [b])

assert cc[:] == [
    {(2, 0): -2j, (0, 1): 0j},
    {(2, 0): -2j, (0, 1): (1 + 0j)},
    {(2, 0): -2j, (0, 1): 0j},
    {(2, 0): -2j, (0, 1): (2 + 0j)},
]

zero = g.sparse_tensor.contract([v], [b]) + g.sparse_tensor.contract([t, t2], [b]) * (-1)
print(zero[:])
assert zero[:] == [{}, {}, {}, {}]

zero = g.sparse_tensor.contract([v], [b]) + (-1) * g.sparse_tensor.contract([t, t2], [b])
assert zero[:] == [{}, {}, {}, {}]

zero = g.sparse_tensor.contract([v], [b]) - g.sparse_tensor.contract([t, t2], [b])
assert zero[:] == [{}, {}, {}, {}]

trace = g.sparse_tensor.contract([v], [b, a, alpha])
g.message(f"Trace: {trace}")

assert trace[:] == [{(): -2j}, {(): (1 - 2j)}, {(): -2j}, {(): (2 - 2j)}]


# test general addition
t1 = g.sparse_tensor.tensor(g.sparse_tensor.basis([a]), 4)

t1[0] = 0.5
t1[2] = 2.0

t2 = g.sparse_tensor.tensor(g.sparse_tensor.basis([b]), 4)

t2[0] = -1.0
t2[1] = -3.5

t = t1 + t2

assert (
    t[:]
    == [
        {
            (1, 0): (-1 + 0j),
            (1, 1): (-3.5 + 0j),
            (0, 2): (0.5 + 0j),
            (0, 0): (-0.5 + 0j),
            (2, 2): (2 + 0j),
            (2, 1): (-1.5 + 0j),
            (0, 1): (-3 + 0j),
            (2, 0): (1 + 0j),
        }
    ]
    * 4
)

trev = t2 + t1

zero = t - trev
assert zero[:] == [{}, {}, {}, {}]

# call global sum
gs = t.global_sum()
