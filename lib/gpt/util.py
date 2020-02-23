#
# GPT
#
# Authors: Christoph Lehner 2020
#

# test if of number type
def isnum(x):
    return isinstance(x, (int, float, complex)) and not isinstance(x, bool)
