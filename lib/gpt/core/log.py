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
import gpt, sys, os, signal, datetime, socket, cgpt
from inspect import getframeinfo, stack

verbose = gpt.default.is_verbose("message_context")
last_context = ""


def message(*a, force_output=False):
    global last_context

    # conversion to string can be an mpi process (i.e. for lattice),
    # so need to do it on all ranks
    s = " ".join([str(x) for x in a])
    if verbose:
        slines = []
        for st in stack()[1:]:
            caller = getframeinfo(st[0])
            slines.append(f"{caller.filename}:{caller.lineno}")
        cpath = os.path.commonpath(slines)
        cs = ";".join([x[len(cpath) + 1 :] for x in slines])
        ctx = f"[{cpath}|{cs}]"
        if ctx != last_context:
            s = f"{ctx}\n{s}"
            last_context = ctx
    if gpt.rank() == 0 or force_output:
        lines = s.split("\n")
        if len(lines) > 0:
            print("GPT : %14.6f s :" % gpt.time(), lines[0])
            for line in lines[1:]:
                print("                       :", line)
        sys.stdout.flush()


