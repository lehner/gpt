#
# GPT
#
# Authors: Christoph Lehner 2020
#
from gpt import rank

def message(*a):
    # conversion to string can be an mpi process (i.e. for lattice),
    # so need to do it on all ranks
    s=[ str(x) for x in a ]
    if rank() == 0:
        print(*s)
