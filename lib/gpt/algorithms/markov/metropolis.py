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
import gpt as g
import numpy


def boltzman_factor(h1, h0):
    return numpy.exp(-h1 + h0)


# Given the probability density P(f(x)), the metropolis algorithm:
# - computes the initial probability P0 = P(f(x0))
# - waits for a proposal for the new fields with w[x1 <- x0]
# - computes the final probability P1 = P(f(x1))
# - performs the accept/reject step, ie accepts with probability = min{1, P(f(x1))/P(f(x0))}
def metropolis(rng, probability_ratio=boltzman_factor):
    def trial(fields):
        # save state
        previous_state = g.copy(fields)

        # get grid for global accept/reject decision
        tmp = fields
        while isinstance(tmp, list):
            tmp = tmp[0]
        grid = tmp.grid

        def accept_reject(f1, f0):
            rr = rng.uniform_real(min=0, max=1)
            rr = grid.globalsum(rr if grid.processor == 0 else 0.0)
            if probability_ratio(f1, f0) >= rr:
                return True
            g.copy(fields, previous_state)
            return False

        return accept_reject

    return trial
