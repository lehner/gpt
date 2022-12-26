#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Test polynomials
#
import gpt as g
import numpy as np

# grid & rng
grid = g.grid([8, 8, 8, 16], g.double)
rng = g.random("rng")

# chebyshev scalar function against operator function
N = 10
low = 0.1
high = 0.78
hard_code_T = {
    5: lambda x: 5.0 * x - 20.0 * x**3.0 + 16.0 * x**5.0,
    8: lambda x: 1.0 - 32.0 * x**2.0 + 160.0 * x**4.0 - 256.0 * x**6.0 + 128.0 * x**8.0,
}

for order in hard_code_T.keys():
    g.message(f"Cheby tests with low = {low}, high = {high}, order = {order}")
    c = g.algorithms.polynomial.chebyshev(low=low, high=high, order=order)

    # check scalar/lattice/hard_coded
    for val in [rng.uniform_real() for i in range(N)]:
        scalar_result = c.eval(val)

        def mul(dst, src):
            dst @= val * src

        lattice_result = g.complex(grid)
        lattice_result[:] = 1
        lattice_result @= c(mul) * lattice_result
        lattice_result = lattice_result[0, 0, 0, 0]

        eps1 = abs(scalar_result - lattice_result) / abs(lattice_result)

        x = (val - 0.5 * (high + low)) / (0.5 * (high - low))
        chebyT_result = hard_code_T[order](x)

        eps2 = abs(chebyT_result - lattice_result) / abs(lattice_result)
        g.message(
            f"c({val}) = {scalar_result} =!= {lattice_result} =!= {chebyT_result} -> eps = {eps1}, {eps2}"
        )
        assert eps1 < 1e-13
        assert eps2 < 1e-13

    # check derivatives
    for val in [rng.uniform_real() for i in range(N)]:
        eps = 1e-7
        result_numerical = (c.eval(val + eps) - c.eval(val - eps)) / 2.0 / eps
        result_exact = c.evalD(val)
        delta = abs(result_exact - result_numerical) / abs(result_exact)
        g.message(f"c'({val}) = {result_numerical} =!= {result_exact} -> eps = {delta}")
        assert delta < eps

    # check function approximation
    def f(x):
        return 1.0 / (x + 0.1)

    c = g.algorithms.polynomial.chebyshev(low=low, high=high, order=30, func=f)
    for val in [rng.uniform_real(min=low, max=high) for i in range(N)]:
        result_cheby_approx = c(val)
        result_exact = f(val)
        eps = abs(result_exact - result_cheby_approx) / abs(result_exact)
        g.message(f"f({val}) = {result_exact} =!= {result_cheby_approx} -> eps = {eps}")
        assert eps < 1e-13

    # finally, check generation of multiple results at the same time
    funcs = [
        lambda x: 1.0 / (x + 0.1),
        lambda x: 1.0 / (x + 0.2),
        lambda x: 1.0 / (x + 0.3),
    ]
    orders = [10, 20, 30]
    c = g.algorithms.polynomial.chebyshev(low=low, high=high, order=orders, func=funcs)
    for val in [rng.uniform_real(min=low, max=high) for i in range(N)]:
        res = c(val)

        def mul(dst, src):
            dst @= val * src

        lattice_result, lattice_input = (
            [g.complex(grid) for i in range(len(orders))],
            g.complex(grid),
        )
        lattice_input[:] = 1
        c(mul).mat(
            lattice_result, lattice_input
        )  # TODO: look at this again after introducing the vector space concept for matrix_operators
        lattice_result = [x[0, 0, 0, 0] for x in lattice_result]

        for i in range(len(orders)):
            ci = g.algorithms.polynomial.chebyshev(
                low=low, high=high, order=orders[i], func=funcs[i]
            )
            resi = ci(val)
            eps = abs(res[i] - resi) / abs(resi)
            assert eps < 1e-13
            eps = abs(res[i] - lattice_result[i]) / abs(resi)
            assert eps < 1e-13
