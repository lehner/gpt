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

def get(tag,default):
    if tag in sys.argv:
        i = sys.argv.index(tag)
        if i+1 < len(sys.argv):
            return sys.argv[i+1]
    return default

def get_float(tag, default = float("nan")):
    res=get(tag,None)
    if not res is None:
        return float(res)
    return default

def get_int(tag, default):
    res=get(tag,None)
    if not res is None:
        return int(res)
    return default

def get_ivec(tag, default):
    res=get(tag,None)
    if not res is None:
        return [ int(x) for x in res.split(".") ]
    return default

# grid and precision
grid = get_ivec("--grid",[4,4,4,4])
precision = { "single" : gpt.single, "double" : gpt.double }[get("--precision","double")]

# IO parameters
max_io_nodes=get_int("--max_io_nodes",256)

# verbosity
verbose_default="io,bicgstab,cg,fgcr,fgmres,mr,irl,power_iteration,checkpointer,deflate,block_operator,random"
verbose_additional="eval"
verbose = get("--verbose",verbose_default).split(",")
verbose_candidates=",".join(sorted((verbose_default + "," + verbose_additional).split(",")))
def is_verbose(x):
    return x in verbose

# help
if "--help" in sys.argv:
    print("--------------------------------------------------------------------------------")
    print(" GPT Help")
    print("--------------------------------------------------------------------------------")
    print("")
    print(" --grid X.Y.Z.T")
    print("")
    print("   sets the default grid for gpt")
    print("")
    print(" --precision single/double")
    print("")
    print("   sets the default precision for gpt")
    print("")
    print(" --verbose opt1,opt2,...")
    print("")
    print("   sets verbosity options.  candidates: %s" % verbose_candidates)
    print("")
    print(" --max_io_nodes n")
    print("")
    print("   set maximal number of simultaneous IO nodes")
    print("--------------------------------------------------------------------------------")
    sys.stdout.flush()
