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
from gpt.qcd.wick.context import fields_context, evaluation_context
from gpt.qcd.wick.expression import expression

tensor = g.sparse_tensor.tensor
basis = g.sparse_tensor.basis


def coordinate_tensor(indices, b, f):
    n_stride = [1]
    for x in indices:
        n_stride.append(n_stride[-1] * x[1])

    t = tensor(basis(b), n_stride[-1])
    t[:] = [
        f([(i // n_stride[j]) % (n_stride[j + 1] // n_stride[j]) for j in range(len(n_stride) - 1)])
        for i in range(n_stride[-1])
    ]
    return t


class wick:
    def __init__(self):
        self.indices = []
        self.coordinates = []
        self.fermions = []

    def index(self, resource, n, nmax):
        if n > 1:
            return tuple([self.index(resource, 1, nmax) for i in range(n)])
        r = (len(resource), nmax)
        resource.append(r)
        return r

    def coordinate(self, n=1):
        return self.index(self.coordinates, n, None)

    def color_index(self, n=1, ncolor=3):
        return self.index(self.indices, n, ncolor)

    def spin_index(self, n=1, nspin=4):
        return self.index(self.indices, n, nspin)

    def lorentz_index(self, n=1, dimensions=4):
        return self.index(self.indices, n, dimensions)

    class field:
        def __init__(self, index, propagators, is_bar):
            self.index = index
            self.is_bar = is_bar

            self.propagators = {}
            for p in propagators:

                if type(propagators[p]) == dict:
                    self.propagators[p] = propagators[p]
                else:
                    p_sc = {}

                    if isinstance(propagators[p], g.lattice):
                        p_s = g.separate_spin(propagators[p])
                        for a, b in p_s:
                            p_sc[(a, b)] = g.separate_color(p_s[a, b])
                    else:
                        Ns = propagators[p].otype.shape[0]
                        Nc = propagators[p].otype.shape[2]
                        p_sc = {
                            (s1, s2): {
                                (c1, c2): propagators[p].array[s1, s2, c1, c2]
                                for c1 in range(Nc)
                                for c2 in range(Nc)
                            }
                            for s1 in range(Ns)
                            for s2 in range(Ns)
                        }

                    self.propagators[p] = p_sc

        def bar(self):
            return wick.field(self.index, self.propagators, not self.is_bar)

        def __call__(self, x, spin_index, color_index):
            def _eval(context, path):

                propagators = context["propagators"]
                propagators_indices = context["propagators_indices"]

                if not self.is_bar:
                    alpha = spin_index
                    a = color_index

                    propagator_alpha_value = (path + "_i0", alpha[1])
                    propagator_a_value = (path + "_i1", a[1])

                    i_alpha = propagators_indices.index(propagator_alpha_value)
                    i_a = propagators_indices.index(propagator_a_value)

                    t = coordinate_tensor(
                        propagators_indices,
                        [alpha, a],
                        lambda i: {(i[i_alpha], i[i_a]): 1.0},
                    )

                else:
                    beta = spin_index
                    b = color_index

                    propagator = [p for p in propagators if p[1] == path]

                    assert len(propagator) == 1

                    field_path = propagator[0][0]
                    propagator_alpha_value = (field_path + "_i0", beta[1])
                    propagator_a_value = (field_path + "_i1", b[1])

                    i_alpha = propagators_indices.index(propagator_alpha_value)
                    i_a = propagators_indices.index(propagator_a_value)

                    sink_coordinate = context[field_path + "_c0"]
                    source_coordinate = x

                    # do we have the coordinate pair
                    if (sink_coordinate, source_coordinate) not in self.propagators:
                        g.message("Unknown coordinate combination!")
                        return tensor(basis([beta, b]))

                    prop = self.propagators[(sink_coordinate, source_coordinate)]
                    # this is still not yet parallelized, should provide a parallel C++
                    # code to fill in a sparse_tensor from dense numpy arrays
                    t = coordinate_tensor(
                        propagators_indices,
                        [beta, b],
                        lambda i: {
                            (beta_value, b_value): prop[(i[i_alpha], beta_value)][(i[i_a], b_value)]
                            for beta_value in range(beta[1])
                            for b_value in range(b[1])
                        },
                    )

                return t

            def _contract(context, path):
                context.register_field(
                    self.index, self.is_bar, path, [x], [spin_index, color_index]
                )

            return expression([spin_index, color_index], _contract, _eval)

    def fermion(self, propagators):
        return wick.field(self.index(self.fermions, 1, -1), propagators, False)

    def epsilon(self, *indices):

        indices = list(indices)

        def _eval(context, path):

            t = tensor(basis(indices), context["n"])
            for idx, sign in g.epsilon(len(indices)):
                t[tuple(idx)] = sign

            return t

        def _contract(context, path):
            pass

        return expression(indices, _contract, _eval)

    class spin_matrix:
        def __init__(self, tensor):
            self.tensor = tensor

        def __call__(self, alpha, beta):

            # order needs to match order in eval tensor
            indices = [alpha, beta]

            def _eval(context, path):
                t = tensor(basis(indices), context["n"])

                for i in range(alpha[1]):
                    for j in range(beta[1]):
                        v = self.tensor.array[i, j]
                        if abs(v) != 0.0:
                            t[i, j] = v

                return t

            def _contract(context, path):
                pass

            return expression(indices, _contract, _eval)

    def sum(self, *arguments):
        expressions = [a for a in arguments if isinstance(a, expression)]
        indices = [a for a in arguments if not isinstance(a, expression)]

        expression_indices = list(set([i for exp in expressions for i in exp.indices]))
        result_indices = [i for i in expression_indices if i not in indices]

        def _eval(context, path):

            ev = [
                exp.evaluate(context, path + "/sum" + str(ii)) for ii, exp in enumerate(expressions)
            ]

            t = g.sparse_tensor.contract(ev, indices)

            return t

        def _contract(context, path):
            for ii, exp in enumerate(expressions):
                exp.contract(context, path + "/sum" + str(ii))

        return expression(result_indices, _contract, _eval)

    def __call__(self, expression, verbose=False, separate_diagrams=False):

        if verbose:
            g.message(f"Open indices: {expression.indices}")

        f_context = fields_context()

        expression.contract(f_context, "")

        if verbose:
            g.message(f"Performing {len(f_context.fields)} wick contractions")

        # fill context with index - value pairs, initialize some to arguments given here
        # perform all wick contractions and iterate through them here, each contraction

        contractions = f_context.contract(verbose)

        if verbose:
            g.message(f"Resulting in {len(contractions)} diagrams")

        results = []
        diag_index = 0
        for sign, propagators in contractions:

            e_context = evaluation_context()
            e_context["propagators"] = propagators

            # set coordinates of propagators
            for propagator in propagators:
                for j, i in enumerate(f_context.coordinate_arguments[propagator[0]]):
                    e_context[propagator[0] + "_c" + str(j)] = i

            # iterate through indices of propagators
            e_context["propagators_indices"] = []
            e_context["n"] = 1
            for propagator in propagators:
                for j, i in enumerate(f_context.index_arguments[propagator[0]]):
                    e_context["propagators_indices"].append((propagator[0] + "_i" + str(j), i[1]))
                    e_context["n"] *= i[1]

            if diag_index % g.ranks() == g.rank():
                r = expression.evaluate(e_context, "")
            else:
                r = tensor(basis(expression.indices), e_context["n"])

            results.append(sign * r)

            # for each propagator do a loop over its spin/color indices following the algorithm below
            # set the index in the context for each field that is part of a propagator; they
            # should then evaluate as below; if a field is not part of the diagram, it should evaluate
            # to zero
            # <u(a) ubar(b)> = D^{-1}_{ab} = sum_{c} \delta_{ac} D^{-1}_{cb}
            # outer sum over c, evaluate field(a) to delta_{ac}, field.bar(b) to D^{-1}_{cb}

        if separate_diagrams:
            return [r.sum().global_sum().tensor_remove() for r in results]
        else:
            res = tensor(basis([]), results[0].n_parallel)
            for r in results:
                res = res + r
            return res.sum().global_sum().tensor_remove()
