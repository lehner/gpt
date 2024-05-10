#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2022
#
import gpt as g
import numpy as np

rng = g.random("test")

n = 8


# first regress binary ops against double versions
def add(a, b):
    return a + b


def mul(a, b):
    return a * b


def div(a, b):
    return a / b


def sub(a, b):
    return a - b


binaries = [add, sub, mul, div]


def bsqrt(a):
    return np.sqrt(a * a)


unaries = [np.real, np.imag, bsqrt, np.abs, np.linalg.norm]


# now devise trivial identities
def comm(a, b):
    return a * b - b * a


def assoc(a, b):
    return (a * b) * a - a * (b * a)


def prod(a, b):
    return (a * b * a) / (a * a * b) - a / a


def unit(a, b):
    return a / a - b / b


def sqrt(a, b):
    return np.sqrt(a * a) * b * np.sqrt(a * a) - a * a * b


def tsum(a, b):
    return (a + b) - a - b


def scale(a, b):
    return ((a + b) * (a + b) - a * a - b * b - 2.0 * a * b) / (abs(a * a) + abs(b * b))


def rscale(a, b):
    return ((a - b) * (a - b) - a * a - b * b + a * b * 2.0) / (abs(a * a) + abs(b * b))


def long(a, b):
    r = 2.0 * a + b * 3.0 - a - a - 3.0 * b + 3.0 * b * a / a / b / 3.0 - 1.0
    r -= a
    r *= b
    r /= b
    r += a
    return r / (abs(a) + abs(b))


def real(a, b):
    return np.real(a) - a


def imag(a, b):
    return np.imag(a)


def tabs(a, b):
    return np.abs(a) - np.sqrt(a * a)


trivialities = [comm, assoc, sqrt, prod, unit, tsum, scale, rscale, long, real, imag, tabs]


#
# qfloat_array tests
#
qa = 1.0 / g.qfloat_array([rng.normal().real for i in range(n)])
qb = 1.0 / g.qfloat_array([rng.normal().real for i in range(n)])
qa_double = qa.leading()
qb_double = qb.leading()

for unary in unaries:
    eps = float(np.linalg.norm(unary(qa) - g.qfloat_array(unary(qa_double))))
    g.message(f"Regress qfloat_array :: {unary.__name__} : {eps} < 1e-14")
    assert eps < 1e-14

for binary in binaries:
    eps = float(np.linalg.norm(binary(qa, qb) - g.qfloat_array(binary(qa_double, qb_double))))
    g.message(f"Regress qfloat_array :: {binary.__name__} : {eps} < 1e-14")
    assert eps < 1e-14

for binary in trivialities:
    eps = float(np.linalg.norm(binary(qa, qb)))
    epsr = float(np.linalg.norm(binary(qa_double, qb_double)))
    g.message(
        f"Test qfloat_array :: {binary.__name__} : {eps} < 1e-28 (quad precision)    {epsr} < 1e-14 (double precision)"
    )
    assert eps < 1e-28
    assert epsr < 1e-14


#
# qfloat tests
#
qa = 1.0 / g.qfloat(rng.normal().real)
qb = 1.0 / g.qfloat(rng.normal().real)
qa_double = qa.leading()
qb_double = qb.leading()

for unary in unaries:
    eps = float(np.linalg.norm(unary(qa) - g.qfloat(unary(qa_double))))
    g.message(f"Regress qfloat :: {unary.__name__} : {eps} < 1e-14")
    assert eps < 1e-14

for binary in binaries:
    eps = float(np.linalg.norm(binary(qa, qb) - g.qfloat(binary(qa_double, qb_double))))
    g.message(f"Regress qfloat :: {binary.__name__} : {eps} < 1e-14")
    assert eps < 1e-14

for binary in trivialities:
    eps = float(np.linalg.norm(binary(qa, qb)))
    epsr = float(np.linalg.norm(binary(qa_double, qb_double)))
    g.message(
        f"Test qfloat :: {binary.__name__} : {eps} < 1e-28 (quad precision)    {epsr} < 1e-14 (double precision)"
    )
    assert eps < 1e-28
    assert epsr < 1e-14


#
# qcomplex tests
#
qa = 1.0 / g.qcomplex(rng.cnormal())
qb = 1.0 / g.qcomplex(rng.cnormal())
qa_double = qa.leading()
qb_double = qb.leading()

binaries = [add, sub, mul, div]
unaries = [np.real, np.imag, np.abs]


def creal(a, b):
    return np.real(a) - a.real


def cimag(a, b):
    return np.imag(a) - a.imag


trivialities = [comm, assoc, prod, unit, tsum, scale, rscale, long, creal, cimag]

for unary in unaries:
    eps = float(abs(g.qcomplex(unary(qa)) - g.qcomplex(unary(qa_double))))
    g.message(f"Regress qcomplex :: {unary.__name__} : {eps} < 1e-14")
    assert eps < 1e-14

for binary in binaries:
    eps = float(abs(binary(qa, qb) - g.qcomplex(binary(qa_double, qb_double))))
    g.message(f"Regress qcomplex :: {binary.__name__} : {eps} < 1e-14")
    assert eps < 1e-14

for binary in trivialities:
    eps = float(abs(g.qcomplex(binary(qa, qb))))
    epsr = float(abs(binary(qa_double, qb_double)))
    g.message(
        f"Test qcomplex :: {binary.__name__} : {eps} < 1e-28 (quad precision)    {epsr} < 1e-14 (double precision)"
    )
    assert eps < 1e-28
    assert epsr < 1e-14


#
# qcomplex_array tests
#
qa = 1.0 / g.qcomplex_array([rng.cnormal() for i in range(n)])
qb = 1.0 / g.qcomplex_array([rng.cnormal() for i in range(n)])
qa_double = qa.leading()
qb_double = qb.leading()

binaries = [add, sub, mul, div]
unaries = [np.real, np.imag, np.abs]
trivialities = [comm, assoc, prod, unit, tsum, scale, rscale, long, creal, cimag]

for unary in unaries:
    eps = float(np.linalg.norm(g.qcomplex_array(unary(qa)) - g.qcomplex_array(unary(qa_double))))
    g.message(f"Regress qcomplex_array :: {unary.__name__} : {eps} < 1e-14")
    assert eps < 1e-14

for binary in binaries:
    eps = float(np.linalg.norm(binary(qa, qb) - g.qcomplex_array(binary(qa_double, qb_double))))
    g.message(f"Regress qcomplex_array :: {binary.__name__} : {eps} < 1e-14")
    assert eps < 1e-14

for binary in trivialities:
    eps = float(np.linalg.norm(g.qcomplex_array(binary(qa, qb))))
    epsr = float(np.linalg.norm(binary(qa_double, qb_double)))
    g.message(
        f"Test qcomplex :: {binary.__name__} : {eps} < 1e-28 (quad precision)    {epsr} < 1e-14 (double precision)"
    )
    assert eps < 1e-28
    assert epsr < 1e-14


#
# global sum tests
#
grid = g.grid([16, 16, 16, 32], g.double_quadruple)

num_ranks_real = grid.globalsum(1.0)
num_ranks_complex = grid.globalsum(1.0 + 2.0j)

assert isinstance(num_ranks_real, g.qfloat)
assert isinstance(num_ranks_complex, g.qcomplex)

assert int(float(num_ranks_real)) == grid.Nprocessors
assert int(float(num_ranks_complex.real)) == grid.Nprocessors
assert int(float(num_ranks_complex.imag)) == 2 * grid.Nprocessors

num_ranks_real = grid.globalsum(np.array([1.0, 1.0]))
num_ranks_complex = grid.globalsum(np.array([1.0 + 3.0j, 1.0 + 4.0j]))

assert isinstance(num_ranks_real, g.qfloat_array)
assert isinstance(num_ranks_complex, g.qcomplex_array)

for i in range(2):
    assert int(float(num_ranks_real[i])) == grid.Nprocessors
    assert int(float(num_ranks_complex.real[i])) == grid.Nprocessors
assert int(float(num_ranks_complex.imag[0])) == 3 * grid.Nprocessors
assert int(float(num_ranks_complex.imag[1])) == 4 * grid.Nprocessors
