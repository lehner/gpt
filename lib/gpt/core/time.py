#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann (daniel.richtmann@ur.de)
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
import cgpt
import gpt

t0 = cgpt.time()


def time():
    return cgpt.time() - t0


def iadd(a, b):
    for x in b:
        if x not in a:
            a[x] = 0.0
        a[x] += b[x]


class timer:
    def __init__(self, name, enabled=True):
        self.dt = {"total": 0.0}
        self.f = {}
        self.b = {}
        self.name = name
        self.active = False
        self.current = None
        self.enabled = enabled

    def __iadd__(self, other):
        if isinstance(other, dict):
            for key in other:
                if key not in self.dt:
                    self.dt[key] = 0.0
                    self.b[key] = 0.0
                    self.f[key] = 0.0
                dt = other[key]["time"]
                self.dt[key] += dt
                self.dt["total"] += dt
        else:
            iadd(self.dt, other.dt)
            iadd(self.b, other.b)
            iadd(self.f, other.f)
        return self

    def __call__(self, which=None, flop=None, byte=None):
        """
        first started timer also starts total timer
        with argument which given starts a new timer and ends a previous one if running
        without argument which ends current + total timer
        """

        if not self.enabled:
            return

        if self.active is False and which is not None:
            self.active = True
            self.dt["total"] -= time()

        if self.current is not None:
            self.dt[self.current] += time()
            self.current = None

        if which is not None:
            if which not in self.dt:
                self.dt[which] = 0.0
                self.f[which] = 0.0
                self.b[which] = 0.0
            self.current = which
            self.f[which] += flop if flop is not None else 0.0
            self.b[which] += byte if byte is not None else 0.0
            self.dt[which] -= time()
        elif self.active:
            self.dt["total"] += time()
            self.active = False

    @property
    def total(self):
        return self.dt["total"]

    def __str__(self):
        assert not self.active
        dtp, fp, bp = self.create_print_arrays()

        s = ""

        if dtp["total"] == 0.0:
            return "No time spent here"

        for k, v in sorted(dtp.items(), key=lambda x: x[1]):
            frac = v / dtp["total"] * 100
            if fp["total"] != 0.0 or bp["total"] != 0.0:
                s += f"{self.name}: profiling: {k:20s} = {v:e} s (= {frac:6.2f} %) {fp[k]/v:e} F/s {bp[k]/v:e} B/s\n"
            else:
                s += f"{self.name}: timing: {k:20s} = {v:e} s (= {frac:6.2f} %)\n"

        return s[:-1]

    def create_print_arrays(self):
        """
        Return copies with enhancements we don't want to have in the raw collected data
        """
        dtp, fp, bp = self.dt.copy(), self.f.copy(), self.b.copy()

        if "total" in dtp:
            total = dtp["total"]
            profiled = sum(dtp.values()) - total
            dtp["unprofiled"] = total - profiled
        else:
            dtp["total"] = sum(dtp.values())
            dtp["unprofiled"] = 0.0  # by construction

        fp["total"] = sum(fp.values())
        bp["total"] = sum(bp.values())
        fp["unprofiled"] = 0.0
        bp["unprofiled"] = 0.0

        return dtp, fp, bp
