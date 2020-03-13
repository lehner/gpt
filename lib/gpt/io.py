#
# GPT
#
# Authors: Christoph Lehner 2020
#
import cgpt
import gpt

def load(*a):
    r=cgpt.load(*a, "io" in gpt.default.verbose)
    print(r)

