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


verbose_profile = None


def profile_reset():
    if verbose_profile is not None:
        verbose_profile.reset()


def profile_save(fn):
    if verbose_profile is not None:
        verbose_profile.save(fn)


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


class profiler_summary:
    def __init__(self, dt=1.0, max_summarize=20):
        self.dt = dt
        self.max_summarize = max_summarize
        self.reset()

    def reset(self):
        self.tags = {}
        self.current = []
        self.t0 = gpt.time() + self.dt

    def __call__(self, start, tag):

        if start == 1:
            self.current.append(tag)
            path = "/".join(self.current)
            if path not in self.tags:
                tg = [0.0, None, 0.0, 0.0, 1]
                self.tags[path] = tg
            else:
                tg = self.tags[path]
            assert tg[1] is None
            tg[1] = gpt.time()

        elif len(self.current) > 0:
            path = "/".join(self.current)
            tg = self.tags[path]
            assert tg[1] is not None
            dt = gpt.time() - tg[1]
            tg[1] = None
            if tg[0] == 0.0:
                tg[2] = dt
                tg[3] = dt
                tg[4] = 1
            else:
                tg[2] = min(tg[2], dt)
                tg[3] = max(tg[3], dt)
                tg[4] += 1
            tg[0] += dt
            last = self.current.pop()
            assert last == tag

        t1 = gpt.time()
        if t1 > self.t0:
            gpt.message(self.__str__()[0:-1])
            self.t0 = t1 + self.dt

    def save(self, fn):
        f = open(fn, "wt")
        f.write(self.__str__())
        f.close()

    def __str__(self):
        t1 = gpt.time()
        path = "/".join(self.current)

        closed_tags = {
            tg: (
                self.tags[tg][0]
                if self.tags[tg][1] is None
                else self.tags[tg][0] + t1 - self.tags[tg][1]
            )
            for tg in self.tags
        }

        sorted_tags = sorted(list(closed_tags), key=lambda x: -closed_tags[x])
        total_timed = sum([closed_tags[x] for x in closed_tags if "/" not in x])

        sep = "----------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
        ret = ""
        ret += sep
        ret += f" Profile Summary with {total_timed:g} s timed and {gpt.time() - total_timed:g} s untimed so far\n"
        ret += sep
        ret += " Total                     Path                                                                     Min        Mean       Max           Count\n"
        ret += sep

        def find_path(ctx, p):
            for c in ctx:
                if c[0] == p:
                    return c[1]
            return None

        display_candidates = sorted_tags[0 : self.max_summarize]
        to_display = []
        for tag in display_candidates:
            path = tag.split("/")
            context = to_display
            for lvl, p in enumerate(path):
                ip = find_path(context, p)
                if ip is None:
                    context.append((p, []))
                    context = context[-1][1]
                else:
                    context = ip

        def walk(current, paths, ret):
            mark = ["+", "*", "-", ">", "#", "$"]
            for ctag, children in paths:
                ccurrent = current + [ctag]

                etag = "  " * (len(current)) + mark[len(current) % len(mark)] + " " + ctag
                tag = "/".join(ccurrent)
                tg = self.tags[tag]

                base = f" {closed_tags[tag] / total_timed * 100:6.2f} %    {closed_tags[tag]:.2e} s    {etag}"
                ret += (
                    base
                    + " " * (100 - len(base))
                    + f"{tg[2]:.2e} / {tg[0] / tg[4]:.2e} / {tg[3]:.2e} s    {tg[4]}"
                    + "\n"
                )
                ret = walk(ccurrent, children, ret)

            return ret

        ret = walk([], to_display, ret)
        ret += sep

        return ret


class timer:
    def __init__(self, name="", enabled=True):
        global verbose_profile
        self.name = name
        self.enabled = enabled
        self.reset()
        if verbose_profile is None:
            if gpt.default.is_verbose("profile"):
                verbose_profile = cgpt.profile_range
            elif gpt.default.is_verbose("profile_summary"):
                verbose_profile = profiler_summary(
                    dt=gpt.default.get_float("--profile_period", 1.0),
                    max_summarize=gpt.default.get_int("--profile_max", 20)
                )
        if verbose_profile is not None:
            self.enabled = True

    def __del__(self):
        self.__call__()

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
            if verbose_profile is not None:
                verbose_profile(0, f"{self.name}::{self.current}")

            self.time[self.current].commit()
            self.current = None

        if which is not None:
            if verbose_profile is not None:
                verbose_profile(1, f"{self.name}::{which}")
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
        nc = max([len(k) for k in dtp] + [20])
        for k, v in reversed(sorted(dtp.items(), key=lambda x: x[1])):
            frac = v / total_time * 100
            tmin = self.time[k].dt_min
            tmax = self.time[k].dt_max
            tavg = self.time[k].dt_sum / self.time[k].n

            s_time = f"{k:{nc}s} {v:.2e} s (= {frac:6.2f} %)"
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
