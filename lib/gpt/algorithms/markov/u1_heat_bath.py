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
#    2020 Tilo Wettig
#    Implementation of heatbath algorithm by Hattori-Nakajima (hep-lat/9210016).
#    Tested against low- and high-temperature expansions in Lautrup-Nauenberg,
#    Phys. Lett. B 95 (1980) 63 and against numbers in Table 3 of Azcoiti et al.,
#    JHEP 08 (2009) 008 (arXiv:0905.0639)

import gpt as g
import numpy as np
import sys


class u1_heat_bath:
    @g.params_convention()
    def __init__(self, rng, params):
        self.rng = rng
        self.params = params

    def __call__(self, link, staple, mask):
        """
        Generate new U(1) links with P(U) = e^{ Re Staple U }
        using the heatbath algorithm of Hattori-Nakajima (hep-lat/9210016),
        which draws a random variable x in (-pi, -pi) from P(x) ~ exp(a cos(x)).
        """
        verbose = g.default.is_verbose(
            "u1_heat_bath"
        )  # need verbosity categories [ performance, progress ]
        assert isinstance(link, g.lattice) and isinstance(staple, g.lattice)

        # component-wise functions needed below
        exp = g.component.exp
        log = g.component.log
        sqrt = g.component.sqrt
        cos = g.component.cos
        tan = g.component.tan
        atan = g.component.atan
        cosh = g.component.cosh
        tanh = g.component.tanh
        atanh = g.component.atanh
        inv = g.component.inv

        # functions needed in Hattori-Nakajima method
        def gmax(x, y):
            return g.where(x > y, x, y)

        def gmin(x, y):
            return g.where(x < y, x, y)

        def h(x):
            return g.eval(
                2.0 * inv(alpha) * atanh(g.eval(beta_s * tan(g.eval((2.0 * x - one) * tmp))))
            )

        def gg(x):
            return exp(g.eval(-a * G(h(x))))

        def G(x):
            return g.eval(
                one
                - cos(x)
                - a_inv
                * log(g.eval(one + (cosh(g.eval(alpha * x)) - one) * inv(g.eval(one + beta))))
            )

        # temporaries
        a = g.component.abs(staple)  # absolute value of staple
        a_inv = g.eval(inv(a))  # needed several times
        grid = a.grid
        one = g.identity(g.complex(grid))
        zero = g.identity(g.complex(grid))
        zero[:] = 0
        Unew = g.complex(grid)  # proposal for new links
        accepted = g.complex(grid)  # mask for accepted links
        num_sites = round(g.norm2(g.where(mask, one, zero)))
        x1 = g.complex(grid)
        x2 = g.complex(grid)
        nohit = 0  # to compute acceptance ratio

        # parameters of Hattori-Nakajima method
        eps = 0.001
        astar = 0.798953686083986
        amax = gmax(zero, g.eval(a - astar * one))
        delta = g.eval(0.35 * amax + 1.03 * sqrt(amax))
        alpha = gmin(sqrt(g.eval(a * (2.0 - eps))), gmax(sqrt(g.eval(eps * a)), delta))
        beta = g.eval(
            (
                gmax(
                    g.eval(alpha * alpha * a_inv),
                    g.eval(
                        (cosh(g.eval(np.pi * alpha)) - one)
                        * inv(g.eval(exp(g.eval(2.0 * a)) - one))
                    ),
                )
                - one
            )
        )
        beta_s = sqrt(g.eval((one + beta) * inv(g.eval(one - beta))))
        tmp = atan(g.eval(inv(beta_s) * tanh(g.eval(0.5 * np.pi * alpha))))

        # main loop (large optimization potential but not time-critical anyway)
        num_accepted = 0
        accepted[:] = 0
        Unew[:] = 0
        # worst-case acceptance ratio of Hattori-Nakajima is 0.88
        while num_accepted < num_sites:
            self.rng.uniform_real(x1, min=0.0, max=1.0)
            self.rng.uniform_real(x2, min=0.0, max=1.0)
            Unew = g.where(accepted, Unew, exp(g.eval(1j * h(x1))))
            newly_accepted = g.where(x2 < gg(x1), one, zero)
            accepted = g.where(mask, g.where(newly_accepted, newly_accepted, accepted), zero)
            num_accepted = round(g.norm2(g.where(accepted, one, zero)))
            nohit += num_sites - num_accepted

        if verbose:
            g.message(f"Acceptance ratio for U(1) heatbath update = {num_sites/(num_sites+nohit)}")

        # Unew was drawn with phase angle centered about zero
        # -> need to shift this by phase angle of staple
        # (we update every link, thus accepted = mask)
        link @= g.where(accepted, Unew * staple * a_inv, link)
