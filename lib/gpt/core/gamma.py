#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt

class gamma_base:
    def __init__(self, gamma):
        self.gamma = gamma

    def __mul__(self, other):
        return gpt.expr(self).__mul__(other)

gamma = {
    0 : gamma_base(0),
    1 : gamma_base(1),
    2 : gamma_base(2),
    3 : gamma_base(3),
    5 : gamma_base(4),
    "X" : gamma_base(0),
    "Y" : gamma_base(1),
    "Z" : gamma_base(2),
    "T" : gamma_base(3),
}
