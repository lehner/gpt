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
import os, inspect, gpt, cgpt

class params_convention:

    # Allows for definition of default parameters and allows for
    # convenient calling with combination of dict (as, e.g., from
    # gpt.params("txt")) and kwargs.  params must be last parameter
    # and a dictionary in receiving function.
    #
    # Guidelines:
    #
    # - Avoid defining default arguments if possible, it is better
    #   to force the user to be aware of the chosen parameters.
    #
    # - If you define default parameters, make them a conservative
    #   choice, e.g., stopping conditions to machine precision.
    #
    # - A good candidate for default parameters is optional uncritical
    #   behavior such as additional reporting or memory advices.
    def __init__(self, default = {}, **kwdefault):
        self.default = {**default, **kwdefault}

    def __call__(self, f):
        fparams = inspect.signature(f).parameters

        # Get last defined parameter (which should be the params dict)
        last_fparam = next(reversed(fparams.values()))

        # If an annotation is of the last parameter is given, it should be a dict
        assert (
            last_fparam.annotation == inspect._empty or last_fparam.annotation == dict
        )

        # Last argument is params
        nargs = len(fparams) - 1

        def wrap(*args, **kwargs):
            assert(len(args) >= nargs)
            positional = args[: nargs]
            params = {**{k: v for d in args[nargs :] for k, v in d.items()}, **kwargs}
            for p,v in self.default.items():
                if not p in params:
                    params[p]=v
            return f(*positional, params)

        return wrap


def params(fn, verbose = False):
    fn=os.path.expanduser(fn)
    dat=open(fn).read()
    if verbose:
        gpt.message("********************************************************************************")
        gpt.message("   Load %s:" % fn)
        gpt.message("********************************************************************************\n%s" % dat.strip())
        gpt.message("********************************************************************************")
    r=eval(dat,globals())
    assert( type(r) == dict )
    return r
