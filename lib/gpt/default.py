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

def get_all(tag,default):
    res=[]
    for i,x in enumerate(sys.argv):
        if x == tag:
            if i+1 < len(sys.argv):
                res.append(sys.argv[i+1])
    if len(res) == 0:
        res=[ default ]
    return res

def get_single(tag,default):
    r=get_all(tag,default)
    assert(len(r) == 1)
    return r[0]

def get_float(tag, default = float("nan")):
    res=get_single(tag,None)
    if not res is None:
        return float(res)
    return default

def get_int(tag, default):
    res=get_single(tag,None)
    if not res is None:
        return int(res)
    return default

def get_ivec(tag, default, ndim):
    res=get_all(tag,None)
    if res[0] is None:
        return default
    for x in res:
        v=[ int(y) for y in x.split(".") ]
        if len(v) == ndim:
            return v
    return default

# Alias
get=get_single

# IO parameters
max_io_nodes=get_int("--max_io_nodes",256)

# verbosity
verbose_default="io,bicgstab,cg,dci,fgcr,fgmres,mr,irl,power_iteration,checkpointer,deflate,block_operator,random"
verbose_additional="eval,merge"
verbose = get_single("--verbose",verbose_default).split(",")
verbose_candidates=",".join(sorted((verbose_default + "," + verbose_additional).split(",")))
def is_verbose(x):
    return x in verbose

# help
if "--help" in sys.argv:
    print("--------------------------------------------------------------------------------")
    print(" GPT Help")
    print("--------------------------------------------------------------------------------")
    print("")
    print(" --mpi X.Y.Z.T")
    print("")
    print("   Set the mpi layout for four-dimensional grids.")
    print("   The layout for other dimensions can be specified")
    print("   by additional parameters such as --mpi X.Y.Z for")
    print("   the mpi layout for three-dimensional grids.")
    print("")
    print(" --verbose opt1,opt2,...")
    print("")
    print("   Set verbosity options.")
    print("   Candidates: %s" % verbose_candidates)
    print("")
    print(" --max_io_nodes n")
    print("")
    print("   Set maximal number of simultaneous IO nodes.")
    print("--------------------------------------------------------------------------------")
    sys.stdout.flush()
