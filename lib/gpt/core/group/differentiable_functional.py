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
import numpy as np

approximation_scheme_4 = [
    (-1.0 / 12.0, +2.0),
    (+2.0 / 3.0, +1.0),
    (-2.0 / 3.0, -1.0),
    (+1.0 / 12.0, -2.0),
]


def _scaled(c, w):
    # list-aware scalar scale: a single field may itself be a list of
    # lattices (e.g. the 4 gauge links of a gauge field node).  Materialize
    # to lattices so g.group.compose (which does g(left)) sees plain fields.
    if isinstance(w, list):
        return [_scaled(c, x) for x in w]
    return g(c * w)


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
        # the weight of each field (fields are found by identity: equal
        # numbers are different parameters)
        w = [next((x for d, x in zip(dfields, weights) if d is f), None) for f in fields]
        return sum(
            [
                (cc / epsilon)
                * self(
                    [
                        f if wf is None else g(g.group.compose(_scaled(dd * epsilon, wf), f))
                        for f, wf in zip(fields, w)
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
        # relative error (absolute for a vanishing difference quotient, e.g.
        # a functional that is constant here); a non-finite error fails
        eps = abs(a - b) / abs(b) if b != 0 else abs(a - b)
        g.message(f"Assert gradient error: {eps} < {epsilon_assert}")
        if not eps <= epsilon_assert:
            g.message(f"Error: gradient = {a} <> approximate_gradient = {b}")
            assert False
        # the gradient needs to live in cartesian.  A single field may itself
        # be a list of lattices (e.g. a gauge-field node), so inspect element-wise
        for gr, ww in zip(gradient, weights):
            for gr_i, ww_i in zip(g.util.to_list(gr), g.util.to_list(ww)):
                if g.util.is_num(gr_i) or isinstance(gr_i, np.ndarray):
                    continue  # complex additive, no otype
                if gr_i.otype.__name__ != ww_i.otype.__name__:
                    g.message(
                        f"Gradient has incorrect object type: {gr_i.otype.__name__} != {ww_i.otype.__name__}"
                    )
                eps = g.group.defect(gr_i)
                if eps > epsilon_assert:
                    g.message(f"Error: cartesian defect: {eps} > {epsilon_assert}")
                    assert False

    def transformed(self, t, indices=None, projection=None):
        return transformed(self, t, indices, projection)

    def __add__(self, other):
        return added(self, other)

    def __radd__(self, other):
        # called if not isinstance(other, differentiable_functional)
        # needed to make sum([ f1, f2, ... ]) work
        assert other == 0
        return self

    def __mul__(self, other):
        return scaled(other, self)

    def __rmul__(self, other):
        return scaled(other, self)


class added(differentiable_functional):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, fields):
        a = self.a(fields)
        b = self.b(fields)
        # g.message("Action",a,b)
        return a + b

    def gradient(self, fields, dfields):
        a_grad = self.a.gradient(fields, dfields)
        b_grad = self.b.gradient(fields, dfields)
        assert len(a_grad) == len(b_grad)
        for i, (x, y) in enumerate(zip(a_grad, b_grad)):
            if isinstance(x, g.lattice):
                x += y
            else:
                # numbers and arrays: x += y would only rebind x
                a_grad[i] = x + y
        return a_grad


class scaled(differentiable_functional):
    def __init__(self, s, f):
        self.s = s
        self.f = f

    def __call__(self, fields):
        return self.s * self.f(fields)

    def gradient(self, fields, dfields):
        grad = self.f.gradient(fields, dfields)
        return [g(self.s * x) for x in grad]


class transformed(differentiable_functional):
    def __init__(self, f, t, indices, projection):
        self.f = f
        self.t = t
        self.indices = indices
        self.projection = projection
        # apply t only to indices, after applying t, keep only projection

    def __call__(self, fields):
        indices = self.indices if self.indices is not None else range(len(fields))
        projection = self.projection if self.projection is not None else range(len(fields))
        fields_indices = [fields[i] for i in indices]
        fields_transformed = self.t(fields_indices)
        fields_prime = [
            fields_transformed[indices.index(i)] if i in indices else fields[i] for i in projection
        ]
        return self.f(fields_prime)

    def gradient(self, fields, dfields):

        # save indices w.r.t. which we want the gradients
        derivative_indices = [g.util.index_by_identity(fields, d) for d in dfields]

        # do the forward pass
        indices = self.indices if self.indices is not None else range(len(fields))
        projection = self.projection if self.projection is not None else range(len(fields))
        fields_indices = [fields[i] for i in indices]
        fields_transformed = self.t(fields_indices)
        fields_prime = [
            fields_transformed[indices.index(i)] if i in indices else fields[i] for i in projection
        ]

        # start the backwards pass with a calculation of the gradient with the transformed fields
        gradient_prime = self.f.gradient(fields_prime, fields_prime)

        src_gradient = [None] * len(fields_transformed)
        for i in indices:
            j = indices.index(i)
            if j < len(gradient_prime) and i < len(src_gradient):
                src_gradient[i] = gradient_prime[j]
        for i in range(len(fields_transformed)):
            if src_gradient[i] is None:
                src_gradient[i] = g.group.cartesian(fields_transformed[i])
                if isinstance(src_gradient[i], g.lattice):
                    src_gradient[i][:] = 0

        # now apply the jacobian to the transformed gradients
        gradient_transformed = self.t.jacobian(fields_indices, fields_transformed, src_gradient)

        return [
            gradient_transformed[indices.index(i)] if i in indices else gradient_prime[i]
            for i in derivative_indices
        ]


class directional_derivative(differentiable_functional):
    """Q(fields) = <v, grad_along S> = d/dt S(exp(t v) fields_along) at t = 0,
    the derivative of S along the cartesian direction v (a list, one element
    per field index in along; v is fixed, not a field of Q).

    The gradient uses the symmetry of second derivatives: for any field x,
    grad_x Q = d/de grad_x S(exp(e v) fields_along) at e = 0, computed by a
    finite difference (scheme, step epsilon; error O(epsilon^4) for the
    default) of S.gradient.  For a field along a non-abelian group the flows
    along v and along the gradient direction do not commute, which adds
    -i [v, grad S] (SU(N) fundamental; hermitian algebra elements)."""

    def __init__(self, S, v, along, epsilon=1e-3, scheme=approximation_scheme_4):
        assert len(v) == len(along)
        self.S, self.v, self.along = S, v, along
        self.epsilon, self.scheme = epsilon, scheme

    def _flowed(self, fields, t):
        fields = list(fields)
        for i, v in zip(self.along, self.v):
            fields[i] = g(g.group.compose(_scaled(t, v), fields[i]))
        return fields

    def __call__(self, fields):
        F = self.S.gradient(fields, [fields[i] for i in self.along])
        return g.group.inner_product(self.v, F)

    def gradient(self, fields, dfields):
        fields = g.util.to_list(fields)
        indices = [g.util.index_by_identity(fields, d) for d in g.util.to_list(dfields)]
        result = None
        for cc, dd in self.scheme:
            flowed = self._flowed(fields, dd * self.epsilon)
            gr = self.S.gradient(flowed, [flowed[i] for i in indices])
            c = cc / self.epsilon
            result = [g(c * x) for x in gr] if result is None else [g(r + c * x) for r, x in zip(result, gr)]
        # commutator correction for fields along a non-abelian group
        along = [i for i in indices if i in self.along]
        if along:
            F = self.S.gradient(fields, [fields[i] for i in along])
            for i, f in zip(along, F):
                otype = fields[i].otype
                if otype.cartesian().__name__ == otype.__name__ or getattr(otype, "Nc", 1) == 1:
                    continue  # additive or abelian
                if not otype.__name__.startswith("ot_matrix_su_n_fundamental_group"):
                    raise NotImplementedError(f"directional_derivative along {otype.__name__}")
                v = self.v[self.along.index(i)]
                k = indices.index(i)
                result[k] = g(result[k] - 1j * (v * f - f * v))
                result[k].otype = f.otype
        return result
