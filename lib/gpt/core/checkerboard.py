#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt
import gpt

class even:
    tag=0

class odd:
    tag=1

def pick_cb(cb, dst, src):
    cgpt.lattice_pick_checkerboard(cb.tag,src.obj,dst.obj)

def set_cb(dst, src):
    cgpt.lattice_set_checkerboard(src.obj,dst.obj)

def change_cb(dst, cb):
    assert(dst.grid.cb == gpt.redblack)
    cgpt.lattice_change_checkerboard(dst.obj,cb.tag)
