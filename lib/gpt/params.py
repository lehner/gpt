#
# GPT
#
# Authors: Christoph Lehner 2020
#
import os
import gpt

def params(fn):
    fn=os.path.expanduser(fn)
    r=eval(open(fn).read(),globals())
    assert( type(r) == dict )
    return r

