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
import gpt, cgpt, numpy, sys
from gpt.params import params_convention


class random:
    def __init__(self, first, second=None):

        if type(first) == dict and second is None:
            s = first["seed"]
            engine = first["engine"]
        else:
            s = first
            engine = second
            if engine is None:
                engine = "vectorized_ranlux24_389_64"

        self.verbose = gpt.default.is_verbose("random")
        self.verbose_performance = gpt.default.is_verbose("random_performance")
        t0 = gpt.time()
        self.obj = cgpt.create_random(engine, s)
        t1 = gpt.time()

        if self.verbose:
            gpt.message(
                "Initializing gpt.random(%s,%s) took %g s" % (s, engine, t1 - t0)
            )

    def __del__(self):
        cgpt.delete_random(self.obj)

    def sample(self, t, p):
        if type(t) == list:
            for x in t:
                self.sample(x, p)
            return t
        elif t is None:
            return cgpt.random_sample(self.obj, p)
        elif type(t) == gpt.lattice:
            t0 = gpt.time()
            cgpt.random_sample(self.obj, {**p, **{"lattices": [t]}})
            t1 = gpt.time()
            assert "pos" not in p  # to ensure that deprecated code is not used

            # optimize memory mapping
            t.swap(gpt.copy(t))

            if self.verbose_performance:
                szGB = t.global_bytes() / 1024.0 ** 3.0
                gpt.message(
                    "Generated %g GB of random data at %g GB/s"
                    % (szGB, szGB / (t1 - t0))
                )

            return t
        else:
            assert 0

    @params_convention(mu=0.0, sigma=1.0)
    def normal(self, t=None, p={}):
        return self.sample(t, {**{"distribution": "normal"}, **p})

    @params_convention(mu=0.0, sigma=1.0)
    def cnormal(self, t=None, p={}):
        return self.sample(t, {**{"distribution": "cnormal"}, **p})

    @params_convention(min=0.0, max=1.0)
    def uniform_real(self, t=None, p={}):
        r = self.sample(t, {**{"distribution": "uniform_real"}, **p})
        if t is None:
            r = r.real
        return r

    @params_convention(min=0, max=1)
    def uniform_int(self, t=None, p={}):
        r = self.sample(t, {**{"distribution": "uniform_int"}, **p})
        if t is None:
            r = int(r.real)
        return r

    @params_convention(n=2)
    def zn(self, t=None, p={}):
        return self.sample(t, {**{"distribution": "zn"}, **p})

    @params_convention(scale=1.0)
    def normal_element(self, out, p={}):
        return self.element(out, p, normal=True)

    @params_convention(scale=1.0)
    def uniform_element(self, out, p={}):
        return self.element(out, p, normal=False)

    @params_convention(scale=1.0, normal=False)
    def element(self, out, p={}):

        if type(out) == list:
            return [self.element(x, p) for x in out]

        t = gpt.timer("element")

        scale = p["scale"]
        normal = p["normal"]
        grid = out.grid

        t("complex")
        ca = gpt.complex(grid)
        ca.checkerboard(out.checkerboard())

        t("cartesian_space")
        cartesian_space = gpt.group.cartesian(out)
        t("csset")
        cartesian_space[:] = 0

        t("gen")
        gen = cartesian_space.otype.generators(grid.precision.complex_dtype)
        t()
        for ta in gen:

            t("rng")

            if normal:
                self.normal(ca)
            else:
                self.uniform_real(ca, {"min": -0.5, "max": 0.5})
            t("lc")
            cartesian_space += scale * ca * ta

        t("conv")
        gpt.convert(out, cartesian_space)
        t()

        # gpt.message(t)
        return out

    def choice(self, array, n):
        if isinstance(array, numpy.ndarray):
            return numpy.take(
                array,
                [self.uniform_int(min=0, max=len(array) - 1) for i in range(n)],
                axis=0,
            )
        else:
            return [
                array[self.uniform_int(min=0, max=len(array) - 1)] for i in range(n)
            ]


# sha256
def sha256(mv):
    if type(mv) == memoryview:
        a = cgpt.util_sha256(mv)
        r = a[0]
        for i in range(7):
            r = r * (2 ** 32) + a[1 + i]
        return r
    else:
        return sha256(memoryview(mv))
