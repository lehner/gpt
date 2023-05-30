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
import gpt as g

approximation_scheme_4 = [
    (-1.0 / 12.0, +2.0),
    (+2.0 / 3.0, +1.0),
    (-2.0 / 3.0, -1.0),
    (+1.0 / 12.0, -2.0),
]


class differentiable_functional:
    def __init__(self):
        return

    def __call__(self, fields):
        raise NotImplementedError()

    def gradient(self, fields, dfields):
        raise NotImplementedError()

    def single_field_gradient(inner):
        # can only differentiate with respect to a single argument, but
        # handle both list and non-list arguments
        def f(self, fields, dfields):
            if isinstance(fields, list):
                assert len(fields) == len(dfields) and fields[0] is dfields[0]
                return [inner(self, fields[0])]
            else:
                return inner(self, fields)

        return f

    def multi_field_gradient(inner):
        def f(self, fields, dfields):
            return_list = isinstance(dfields, list)
            r = inner(self, g.util.to_list(fields), g.util.to_list(dfields))
            if not return_list:
                return r[0]
            return r

        return f

    def approximate_gradient(
        self, fields, dfields, weights, epsilon=1e-5, scheme=approximation_scheme_4
    ):
        fields = g.util.to_list(fields)
        dfields = g.util.to_list(dfields)
        weights = g.util.to_list(weights)
        assert len(dfields) == len(weights)
        return sum(
            [
                (cc / epsilon)
                * self(
                    [
                        g(g.group.compose((dd * epsilon) * weights[dfields.index(f)], f))
                        if f in dfields
                        else f
                        for f in fields
                    ]
                )
                for cc, dd in scheme
            ]
        )

    def assert_gradient_error(self, rng, fields, dfields, epsilon_approx, epsilon_assert):
        fields = g.util.to_list(fields)
        dfields = g.util.to_list(dfields)
        weights = rng.normal_element(g.group.cartesian(dfields))
        # the functional needs to be real
        eps = complex(self(fields)).imag
        g.message(f"Test that functional is real: {eps}")
        assert eps == 0.0
        # the gradient needs to be correct
        gradient = self.gradient(fields, dfields)
        a = sum([g.group.inner_product(w, gr) for gr, w in zip(gradient, weights)])
        b = self.approximate_gradient(fields, dfields, weights, epsilon=epsilon_approx)
        eps = abs(a - b) / abs(b)
        g.message(f"Assert gradient error: {eps} < {epsilon_assert}")
        if eps > epsilon_assert:
            g.message(f"Error: gradient = {a} <> approximate_gradient = {b}")
            assert False
        # the gradient needs to live in cartesian
        for gr, ww in zip(gradient, weights):
            if gr.otype.__name__ != ww.otype.__name__:
                g.message(
                    f"Gradient has incorrect object type: {gr.otype.__name__} != {ww.otype.__name__}"
                )
            eps = g.group.defect(gr)
            if eps > epsilon_assert:
                g.message(f"Error: cartesian defect: {eps} > {epsilon_assert}")
                assert False

    def transformed(self, t):
        return transformed(self, t)


class transformed(differentiable_functional):
    def __init__(self, f, t):
        self.f = f
        self.t = t

    def __call__(self, fields):
        return self.f(self.t(fields))

    def gradient(self, fields, dfields):
        indices = [fields.index(d) for d in dfields]

        fields_prime = self.t(fields)

        transformed_fields = [i for i in range(len(fields)) if fields_prime[i] is not fields[i]]

        # for now only accept gradients with respect to all transformed fields
        assert indices == transformed_fields

        gradient_prime = self.f.gradient(fields_prime, [fields_prime[i] for i in indices])

        return self.t.jacobian(fields, fields_prime, gradient_prime)
