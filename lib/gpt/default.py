#
# GPT
#
# Authors: Christoph Lehner 2020
#
import sys

def get_vec(tag, default):
    if tag in sys.argv:
        i = sys.argv.index(tag)
        if i+1 < len(sys.argv):
            return [ int(x) for x in sys.argv[i+1].split(".") ]
    return default

grid = get_vec("--grid",[4,4,4,4])
