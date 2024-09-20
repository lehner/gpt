#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2024  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import json, hashlib, os, sys


runtime_tune_cache = {}


def save_cache(fn, data):
    if g.rank() == 0:
        os.makedirs(".gpt_auto_tune", exist_ok=True)
        fout = open(fn, "wt")
        json.dump(data, fout)
        fout.close()
    runtime_tune_cache[fn] = data


def load_cache(fn):
    if fn not in runtime_tune_cache:
        runtime_tune_cache[fn] = json.load(open(fn, "rt"))
    return runtime_tune_cache[fn]


def auto_tuned_method(method):
    def wrapper(self, *args):
        if not self.at_active:
            return method(self, self.at_default_param, *args)

        if self.at_tuned_params is not None:
            return method(self, self.at_tuned_params["params"], *args)

        # create a snapshot of parameters to restore
        args = list(args)
        args_snapshot = g.copy(args)

        # do experiments
        dt_warmup = -g.time()
        g.message(f"Auto-tune {self.at_tag} warmup")
        method(self, self.at_default_param, *args)
        dt_warmup += g.time()

        g.copy(args, args_snapshot)

        dts = []
        for p in self.at_params:
            dt = -g.time()
            g.message(f"Auto-tune {self.at_tag} with {p}")
            method(self, p, *args)
            dt += g.time()

            g.copy(args, args_snapshot)
            dts.append(dt)

        g.message(f"Tuning result for {self.at_tag}:")
        g.message(f"- Warmup with {self.at_default_param} took {dt_warmup:g} s")
        for i in range(len(dts)):
            g.message(f"- {self.at_params[i]} took {dts[i]:g} s")
        imin = min(range(len(dts)), key=dts.__getitem__)
        imin = g.broadcast(0, imin)
        self.at_tuned_params = {"tag": self.at_tag, "params": self.at_params[imin], "results": dts}
        g.message(f"Tuning result (use {self.at_tuned_params} will be saved in {self.at_fn}")
        save_cache(self.at_fn, self.at_tuned_params)

        g.barrier()

        # run with tuned params
        return method(self, self.at_tuned_params["params"], *args)

    return wrapper


class auto_tuned_class:
    def __init__(self, tag, params, default_param):
        self.at_tag = tag
        self.at_params = params
        self.at_default_param = default_param
        self.at_active = g.default.has("--auto-tune")
        self.at_verbose = g.default.is_verbose("auto_tune")

        hash_tag = str(hashlib.sha256(tag.encode("utf-8")).hexdigest())

        self.at_fn = f".gpt_auto_tune/{hash_tag}.json"
        if g.rank() == 0 and os.path.exists(self.at_fn) and self.at_active:
            try:
                self.at_tuned_params = load_cache(self.at_fn)
                assert self.at_tuned_params["tag"] == tag
                if self.at_verbose:
                    g.message(f"Use tuned results from {self.at_fn}")
            except:
                self.at_tuned_params = {}
        else:
            self.at_tuned_params = {}

        self.at_tuned_params = g.broadcast(0, self.at_tuned_params)

        if len(self.at_tuned_params) == 0:
            self.at_tuned_params = None
