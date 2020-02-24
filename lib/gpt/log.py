#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt, sys

def message(*a):
    # conversion to string can be an mpi process (i.e. for lattice),
    # so need to do it on all ranks
    s=[ str(x) for x in a ]
    if gpt.rank() == 0:
        print(*s)
        sys.stdout.flush()
