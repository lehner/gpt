#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import os
import gpt

class params_convention:
   def __init__(self, nargs):
       self.nargs = nargs
   def __call__(self,f):
       def wrap(*args,**kwargs):
           assert(len(args) >= self.nargs)
           positional=args[:self.nargs]
           params={**{k: v for d in args[self.nargs:] for k, v in d.items()}, **kwargs}
           return f(*positional,params)
       return wrap

def params(fn, verbose = False):
    fn=os.path.expanduser(fn)
    dat=open(fn).read()
    if verbose:
        gpt.message("********************************************************************************")
        gpt.message("   Load %s:" % fn)
        gpt.message("********************************************************************************\n%s" % dat.strip())
        gpt.message("********************************************************************************")
    r=eval(dat,globals())
    assert( type(r) == dict )
    return r
