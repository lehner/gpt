#
# GPT
#
# Authors: Christoph Lehner 2020
#
class complex:
    def __init__(self):
        pass

def mul(a,b):
    if a == b:
        return a
    raise Exception("Unknown type combination " + str(a) + " " + str(b))

