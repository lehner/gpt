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
import sys, gpt


def get_all(tag, default):
    res = []
    for i, x in enumerate(sys.argv):
        if x == tag:
            if i + 1 < len(sys.argv):
                res.append(sys.argv[i + 1])
    if len(res) == 0:
        res = [default]
    return res


def has(tag):
    return tag in sys.argv


def get_single(tag, default):
    r = get_all(tag, default)
    assert len(r) == 1
    return r[0]


def get_float(tag, default=float("nan")):
    res = get_single(tag, None)
    if res is not None:
        return float(res)
    return default


def get_int(tag, default):
    res = get_single(tag, None)
    if res is not None:
        return int(res)
    return default


def get_ivec(tag, default, ndim):
    res = get_all(tag, None)
    if res[0] is None:
        return default
    for x in res:
        v = [int(y) for y in x.split(".")]
        if len(v) == ndim:
            return v
    return default


# Alias
get = get_single

# IO parameters
max_io_nodes = get_int("--max_io_nodes", 256)

# verbosity
verbose_default = (
    "io,bicgstab,cg,defect_correcting,cagcr,fgcr,fgmres,mr,irl,repository,arnoldi,power_iteration,"
    + "checkpointer,modes,random,split,coarse_grid,gradient_descent,adam,non_linear_cg,"
    + "coarsen,qis_map,metropolis,su2_heat_bath,u1_heat_bath,fom,chronological,minimal_residual_extrapolation,"
    + "subspace_minimal_residual"
)
verbose_additional = "eval,merge,orthogonalize,copy_plan"
verbose = set()
verbose_candidates = ",".join(sorted((verbose_default + "," + verbose_additional).split(",")))
verbose_indent = max([len(x) for x in verbose_candidates.split(",")])
verbose_stack = []


def is_verbose(x):
    return x in verbose


def set_verbose(x, status=True):
    if (status is True) and (x not in verbose):
        verbose.add(x)
    if (status is False) and (x in verbose):
        verbose.remove(x)


def push_verbose(x, status):
    verbose_stack.append((x, is_verbose(x)))
    set_verbose(x, status)


def pop_verbose():
    set_verbose(*verbose_stack.pop())


def parse_verbose():
    global verbose
    verbose = set([y for x in get_all("--verbose", verbose_default) for y in x.split(",")])
    for status, mod in [(True, "add"), (False, "remove")]:
        for f in [
            y for x in get_all(f"--verbose_{mod}", None) if x is not None for y in x.split(",")
        ]:
            set_verbose(f, status)


parse_verbose()


# help flag
help_flag = "--help" in sys.argv
if help_flag:
    sys.argv.remove("--help")


def wrap_list(string, separator_split, separator_join, linewidth, indent):
    r = ""
    line = ""
    array = string.split(separator_split)
    for i, word in enumerate(array):
        if len(line) + len(separator_join) + len(word) >= linewidth:
            r = r + "\n" + (" " * indent) + line
            line = ""
        if i + 1 != len(array):
            line = line + word + separator_join
        else:
            line = line + word
    if len(line) != 0:
        r = r + "\n" + (" " * indent) + line
    return r


def process_flags():
    if help_flag:
        gpt.message(
            f"""--------------------------------------------------------------------------------
                 GPT Help
--------------------------------------------------------------------------------

 --mpi X.Y.Z.T

   Set the mpi layout for four-dimensional grids.
   The layout for other dimensions can be specified
   by additional parameters such as --mpi X.Y.Z for
   the mpi layout for three-dimensional grids.

 --verbose opt1,opt2,...

   Set list of verbose operations to opt1,opt2,...

   Candidates:
   {wrap_list(verbose_candidates, ",", ", ", 50, 3)}

   Current selection:
   {wrap_list(",".join(verbose), ",", ", ", 50, 3)}

   For many operations X, refined control is available
   through the options

      X_debug          to display debug messages
      X_convergence    to display algorithmic convergence
      X_performance    to display performance statistics

 --verbose_add opt1,opt2,...

   Add opt1,opt2,... to list of verbose operations.

 --verbose_remove opt1,opt2,...

   Remove opt1,opt2,... from list of verbose operations.

 --max_io_nodes n

   Set maximal number of simultaneous IO nodes.
"""
        )
        sys.exit(0)
