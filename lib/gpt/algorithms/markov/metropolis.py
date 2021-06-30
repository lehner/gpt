#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                        Mattia Bruno
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
import numpy


def boltzman_factor(h1, h0):
    return numpy.exp(-h1 + h0)


# Given the probability density P(f(x)), the metropolis algorithm:
# - computes the initial probability P0 = P(f(x0))
# - makes a proposal for the new fields with w[x1 <- x0]
# - computes the final probability P1 = P(f(x1))
# - performs the accept/reject step, ie accepts with probability = min{1, P(f(x1))/P(f(x0))}
# The class below accepts the following arguments:
# - rng: random number generator
# - w: proposal, a callable function
# - f: the function, argument of the probability density
# - fields: a list of fields
# - prob_ratio: the method to compute the ratio P(f(x1))/P(f(x0)), which implicitly defines P,
#               e.g. boltzman_factor: P(f(x1))/P(f(x0)) = exp(-f(x1) + f(x0))
class metropolis:
    def __init__(self, rng, w, f, fields, prob_ratio=boltzman_factor):
        self.rng = rng
        self.proposal = w
        self.f = f
        self.fields = gpt.core.util.to_list(fields)
        self.prob_ratio = prob_ratio
        tmp = self.fields[0]
        while isinstance(tmp, list):
            tmp = tmp[0]
        self.grid = tmp.grid

    def __call__(self, *vargs):
        previous_fields = gpt.copy(self.fields)
        f0 = self.f(*self.fields)
        self.proposal(*vargs)
        f1 = self.f(*self.fields)

        # decision taken on master node
        accept = (
            1
            if self.prob_ratio(f1, f0)
            >= self.grid.globalsum(
                self.rng.uniform_real(min=0, max=1) if self.grid.processor == 0 else 0.0
            )
            else 0
        )
        if accept == 0:
            gpt.copy(self.fields, previous_fields)

        return [accept, f1 - f0]
