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

    def approximate_gradient(self, fields, dfields, site_weight, epsilon=1e-5):
        return_list = isinstance(fields, list)
        fields, dfields = g.util.to_list(fields), g.util.to_list(dfields)

        a = []

        for x in dfields:
            mu = fields.index(x)

            # vary argument mu
            L = fields[0:mu]
            R = fields[mu + 1 :]

            # This helper function allows for quick checks of gradient implementations on single sites
            c = x.otype.cartesian()
            grid = x.grid
            epsilon = complex(epsilon)

            # move to neutral element of group (\vec{0} in cartesian space)
            t = g.lattice(grid, c)
            t[:] = 0

            # generators of cartesian space
            gen = c.generators(grid.precision.complex_dtype)
            r = gen[0] * complex(0.0)

            # functional at neutral element
            for gg in gen:
                t += epsilon * gg * site_weight
                r += (
                    (
                        self(L + [g(g.group.compose(t, x))] + R)
                        - self(L + [g(g.group.compose(-t, x))] + R)
                    )
                    / (2.0 * epsilon)
                ) * gg
                t -= epsilon * gg * site_weight
            a.append(r)

        if not return_list:
            return a[0]

        return a

    def assert_gradient_error(
        self, rng, fields, dfields, epsilon_approx, epsilon_assert
    ):
        fields, dfields = g.util.to_list(fields), g.util.to_list(dfields)
        test_weight = rng.normal(g.singlet(fields[0].grid))
        gr_val = [g.sum(x * test_weight) for x in self.gradient(fields, dfields)]
        gr_app = self.approximate_gradient(
            fields, dfields, test_weight, epsilon=epsilon_approx
        )
        for mu in range(len(gr_val)):
            a = gr_val[mu]
            b = gr_app[mu]
            if type(a) is complex:
                eps = abs(a - b) / abs(b)
            else:
                eps = (g.norm2(a - b) / g.norm2(b)) ** 0.5
            g.message(f"Assert gradient component {mu} error: {eps} < {epsilon_assert}")
            assert eps < epsilon_assert
