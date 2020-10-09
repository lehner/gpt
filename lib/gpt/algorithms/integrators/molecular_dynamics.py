#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Mattia Bruno
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


class integrator:
    def __init__(self, N, i0, i1):
        self.N = N
        self.i0 = i0
        self.i1 = i1

    def get_act(self):
        return self.i0.get_act() + self.i1.get_act()


class leap_frog(integrator):
    def __call__(self, tau):
        eps = tau / self.N
        verbose = gpt.default.is_verbose("leap_frog")

        time = gpt.timer("leap_frog")
        time("leap_frog")

        self.i0(eps * 0.5)
        for i in range(self.N):
            self.i1(eps)
            if i != self.N - 1:
                self.i0(eps)
        self.i0(eps * 0.5)

        if verbose:
            time()
            gpt.message(f"Leap Frog Integrator ran in {time.dt['total']:g} secs")


class OMF2(integrator):
    def __init__(self, N, i0, i1, l=0.18):
        super().__init__(N, i0, i1)
        self.r0 = l

    def __call__(self, tau):
        eps = tau / self.N
        verbose = gpt.default.is_verbose("omf2")

        time = gpt.timer("OMF4")
        time("OMF4")

        self.i0(self.r0 * eps)
        for i in range(self.N):
            self.i1(eps * 0.5)
            self.i0((1 - 2 * self.r0) * eps)
            self.i1(eps * 0.5)

            if i != self.N - 1:
                self.i0(2.0 * self.r0 * eps)
        self.i0(self.r0 * eps)

        if verbose:
            time()
            gpt.message(f"OMF2 Integrator ran in {time.dt['total']:g} secs")


class OMF4(integrator):
    def __init__(self, N, i0, i1):
        super().__init__(N, i0, i1)
        self.r = [
            0.08398315262876693,
            0.2539785108410595,
            0.6822365335719091,
            -0.03230286765269967,
        ]

    def __call__(self, tau):
        eps = tau / self.N
        f1 = 0.5 - self.r[0] - self.r[2]
        f2 = 1.0 - 2.0 * (self.r[1] + self.r[3])
        verbose = gpt.default.is_verbose("omf4")

        time = gpt.timer("OMF4")
        time("OMF4")

        time("momenta")
        self.i0(self.r[0] * eps)
        for i in range(self.N):
            time("fields")
            self.i1(self.r[1] * eps)
            time("momenta")
            self.i0(self.r[2] * eps)
            time("fields")
            self.i1(self.r[3] * eps)

            time("momenta")
            self.i0(f1 * eps)

            time("fields")
            self.i1(f2 * eps)

            time("momenta")
            self.i0(f1 * eps)

            time("fields")
            self.i1(self.r[3] * eps)
            time("momenta")
            self.i0(self.r[2] * eps)
            time("fields")
            self.i1(self.r[1] * eps)

            if i != self.N - 1:
                time("momenta")
                self.i0(2.0 * self.r[0] * eps)
        time("momenta")
        self.i0(self.r[0] * eps)

        if verbose:
            time()
            gpt.message(time)
            gpt.message(f"OMF4 Integrator ran in {time.dt['total']:g} secs")
