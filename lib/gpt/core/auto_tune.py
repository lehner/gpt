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
import numpy as np
import json, hashlib, os, pickle, sys


runtime_tune_cache = {}

# The collectives of a tuned object run on the communicator of its grid: under a
# split grid (g.split) the groups construct and tune objects independently, so
# world collectives would pair up across groups.  Without a grid (rank-local
# objects, e.g. contraction plans of accelerator buffers) tuning is rank-local.


def _is_root(grid):
    return (grid.processor if grid is not None else g.rank()) == 0


def _broadcast(grid, data):
    if grid is None:
        return data
    root = grid.processor == 0
    buf = pickle.dumps(data) if root else b""
    n = np.array([len(buf)], dtype=np.int64)
    grid.broadcast(0, n)
    a = np.frombuffer(buf, dtype=np.uint8).copy() if root else np.zeros(int(n[0]), dtype=np.uint8)
    grid.broadcast(0, a)
    return pickle.loads(a.tobytes())


def _barrier(grid):
    if grid is not None:
        grid.barrier()


def save_cache(fn, data, grid):
    if _is_root(grid):
        # roots of several groups may write the same file: write a private
        # temporary and rename it (atomic), so readers never see a partial file
        os.makedirs(".gpt_auto_tune", exist_ok=True)
        tmp = f"{fn}.{g.rank()}.tmp"
        with open(tmp, "wt") as fout:
            json.dump(data, fout)
        os.replace(tmp, fn)
    runtime_tune_cache[fn] = data


def load_cache(fn):
    if fn not in runtime_tune_cache:
        runtime_tune_cache[fn] = json.load(open(fn, "rt"))
    return runtime_tune_cache[fn]


def auto_tuned_method(skip_snapshot=False):
    def _method(method):
        def wrapper(self, *args):
            if not self.at_active:
                return method(self, self.at_default_param, *args)

            if self.at_tuned_params is not None:
                return method(self, self.at_tuned_params["params"], *args)

            # create a snapshot of parameters to restore
            args = list(args)
            if not skip_snapshot:
                args_snapshot = g.copy(args)

            # do experiments
            dt_warmup = -g.time()
            g.message(f"Auto-tune {self.at_tag} warmup")
            method(self, self.at_default_param, *args)
            dt_warmup += g.time()

            if not skip_snapshot:
                g.copy(args, args_snapshot)

            dts = []
            for p in self.at_params:
                dt = -g.time()
                g.message(f"Auto-tune {self.at_tag} with {p}")
                method(self, p, *args)
                dt += g.time()

                if not skip_snapshot:
                    g.copy(args, args_snapshot)
                dts.append(dt)

            g.message(f"Tuning result for {self.at_tag}:")
            g.message(f"- Warmup with {self.at_default_param} took {dt_warmup:g} s")
            for i in range(len(dts)):
                g.message(f"- {self.at_params[i]} took {dts[i]:g} s")
            imin = min(range(len(dts)), key=dts.__getitem__)
            imin = _broadcast(self.at_grid, imin)
            self.at_tuned_params = {
                "tag": self.at_tag,
                "params": self.at_params[imin],
                "results": dts,
            }
            g.message(f"Tuning result (use {self.at_tuned_params} will be saved in {self.at_fn}")
            save_cache(self.at_fn, self.at_tuned_params, self.at_grid)

            _barrier(self.at_grid)

            # run with tuned params
            return method(self, self.at_tuned_params["params"], *args)

        return wrapper

    return _method


class auto_tuned_class:
    def __init__(self, tag, params, default_param, grid=None):
        self.at_tag = tag
        self.at_params = params
        self.at_default_param = default_param
        self.at_grid = grid
        self.at_active = g.default.has("--auto-tune")
        self.at_verbose = g.default.is_verbose("auto_tune")

        hash_tag = str(hashlib.sha256(tag.encode("utf-8")).hexdigest())

        self.at_fn = f".gpt_auto_tune/{hash_tag}.json"
        self.at_tuned_params = None
        if not self.at_active:
            return

        tuned = {}
        if grid is None or grid.processor == 0:
            if os.path.exists(self.at_fn):
                try:
                    tuned = load_cache(self.at_fn)
                    assert tuned["tag"] == tag
                    if self.at_verbose:
                        g.message(f"Use tuned results from {self.at_fn}")
                except Exception:
                    tuned = {}

        tuned = _broadcast(grid, tuned)

        if len(tuned) != 0:
            self.at_tuned_params = tuned

    def get_tuned_parameters(self):
        if not self.at_active:
            return self.at_default_param
        if self.at_tuned_params is None:
            return None
        return self.at_tuned_params["params"]

    def save_tuned_parameters(self, params):
        self.at_tuned_params = {"tag": self.at_tag, "params": params, "results": []}
        g.message(f"Tuning result (use {self.at_tuned_params} will be saved in {self.at_fn}")
        save_cache(self.at_fn, self.at_tuned_params, self.at_grid)
