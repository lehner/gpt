#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt, sys
import time as tm

t0=tm.time()

def time():
    return tm.time() - t0

def message(*a):
    # conversion to string can be an mpi process (i.e. for lattice),
    # so need to do it on all ranks
    s=[ str(x) for x in a ]
    if gpt.rank() == 0:
        print("GPT : %14.6f s :" % time(),*s)
        sys.stdout.flush()
