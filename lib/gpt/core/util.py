#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt

# test if of number type
def isnum(x):
    return isinstance(x, (int, float, complex)) and not isinstance(x, bool)

# tensor
def value_to_tensor(val, otype):
    if type(val) == complex:
        return val
    return gpt.tensor(val, otype)

def tensor_to_value(value):
    if type(value) == gpt.tensor:
        value = value.array
    return value


