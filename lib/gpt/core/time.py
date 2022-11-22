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


class timer_component:
    def __init__(self):
        self.dt_sum = 0.0
        self.dt_max = None
        self.dt_min = None
        self.dt_last = None

        self.flop_sum = None
        self.flop_per_sec_max = None
        self.flop_per_sec_min = None
        self.flop_per_sec_last = None

        self.byte_sum = None
        self.byte_per_sec_max = None
        self.byte_per_sec_min = None
        self.byte_per_sec_last = None

        self.n = 0

    def start(self, flop, byte):
        self._t0 = time()
        self._flop = flop
        self._byte = byte

    def commit(self):
        self.register_time_step(time() - self._t0, self._flop, self._byte)

    def register_time_step(self, dt, flop, byte):
        self.dt_max = max([self.dt_max, dt]) if self.dt_max is not None else dt
        self.dt_min = min([self.dt_min, dt]) if self.dt_min is not None else dt
        self.dt_sum += dt
        self.dt_last = dt

        if flop is not None:
            flop_per_sec = flop / dt
            if self.flop_sum is None:
                self.flop_sum = 0.0
            self.flop_sum += flop
            self.flop_per_sec_last = flop_per_sec
            self.flop_per_sec_max = (
                max([self.flop_per_sec_max, flop_per_sec])
                if self.flop_per_sec_max is not None
                else flop_per_sec
            )
            self.flop_per_sec_min = (
                min([self.flop_per_sec_min, flop_per_sec])
                if self.flop_per_sec_min is not None
                else flop_per_sec
            )

        if byte is not None:
            byte_per_sec = byte / dt
            if self.byte_sum is None:
                self.byte_sum = 0.0
            self.byte_sum += byte
            self.byte_per_sec_last = byte_per_sec
            self.byte_per_sec_max = (
                max([self.byte_per_sec_max, byte_per_sec])
                if self.byte_per_sec_max is not None
                else byte_per_sec
            )
            self.byte_per_sec_min = (
                min([self.byte_per_sec_min, byte_per_sec])
                if self.byte_per_sec_min is not None
                else byte_per_sec
            )

        self.n += 1

    def append(self, other):
        self.dt_max = max([self.dt_max, other.dt_max]) if self.dt_max is not None else other.dt_max
        self.dt_min = min([self.dt_min, other.dt_min]) if self.dt_min is not None else other.dt_min
        self.dt_sum += other.dt_sum
        self.dt_last = other.dt_last
        self.n += other.n

    def clone(self):
        c = timer_component()
        c.append(self)
        return c


def iadd(dst, src):
    for x in src:
        if x in dst:
            dst[x].append(src[x])
        else:
            dst[x] = src[x].clone()


class timer:
    def __init__(self, name="", enabled=True):
        self.name = name
        self.enabled = enabled
        self.reset()

    def reset(self):
        self.time = {}
        self.active = False
        self.current = None

    def __iadd__(self, other):
        if isinstance(other, dict):
            for key in other:
                if key not in self.time:
                    self.time[key] = timer_component()
                self.time[key].register_time_step(other[key]["time"], None, None)
        else:
            iadd(self.time, other.time)
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

        if self.current is not None:
            self.time[self.current].commit()
            self.current = None

        if which is not None:
            if which not in self.time:
                self.time[which] = timer_component()
            self.current = which
            self.time[which].start(flop, byte)
        elif self.active:
            self.active = False

    def __str__(self):
        assert not self.active

        # first create statistics
        total_time = 0.0
        dtp = {}
        for tag in self.time:
            total_time += self.time[tag].dt_sum
            dtp[tag] = self.time[tag].dt_sum

        if total_time == 0.0:
            return "No time spent here"

        s = f"{self.name}:\n" if self.name != "" else ""
        for k, v in reversed(sorted(dtp.items(), key=lambda x: x[1])):
            frac = v / total_time * 100
            tmin = self.time[k].dt_min
            tmax = self.time[k].dt_max
            tavg = self.time[k].dt_sum / self.time[k].n

            s_time = f"{k:20s} {v:.2e} s (= {frac:6.2f} %)"
            s += f"{s_time}; time/s = {tmin:.2e}/{tmax:.2e}/{tavg:.2e} (min/max/avg)\n"
            if self.time[k].byte_sum is not None:
                bmin = self.time[k].byte_per_sec_min
                bmax = self.time[k].byte_per_sec_max
                bavg = self.time[k].byte_sum / self.time[k].dt_sum
                s += (" " * len(s_time)) + f"  byte/s = {bmin:.2e}/{bmax:.2e}/{bavg:.2e}\n"
            if self.time[k].flop_sum is not None:
                fmin = self.time[k].flop_per_sec_min
                fmax = self.time[k].flop_per_sec_max
                favg = self.time[k].flop_sum / self.time[k].dt_sum
                s += (" " * len(s_time)) + f"  flop/s = {fmin:.2e}/{fmax:.2e}/{favg:.2e}\n"

        return s[:-1]
