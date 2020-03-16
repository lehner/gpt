#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt
import numpy as np

def point(src, pos):
    src[:]=0
    src[tuple(pos)]=gpt.mspincolor(np.multiply.outer( np.identity(4) , np.identity(3) ))

