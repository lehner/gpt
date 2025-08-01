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


def gradient_flow(U, epsilon, action):
    return g.algorithms.integrator.runge_kutta_4(
        U, lambda Up: [g(-u) for u in action.gradient(Up, Up)], epsilon
    )


action_cache = None
def wilson_flow(U, epsilon):
    global action_cache
    if action_cache is None:
        action_cache = g.qcd.gauge.action.wilson(2.0 * U[0].otype.shape[0])
    return gradient_flow(U, epsilon, action_cache)
