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
from gpt.core import *
from gpt.params import params, params_convention
from gpt.repository import repository
import gpt.default
import gpt.create
import gpt.algorithms
import gpt.qcd
import gpt.qis
import gpt.ml
import gpt.jobs
import socket
import cgpt
import sys
import types

"""
GPT -- Grid Python Toolkit
"""

# initialize cgpt when gpt is loaded
cgpt.init(sys.argv)

# save my hostname
hostname = socket.gethostname()

# process flags
gpt.default.process_flags()

# synonyms
eval = expr_eval

# make module callable
class GPTModule(types.ModuleType):
    def __call__(self, *args):
        return expr_eval(*args)


sys.modules[__name__].__class__ = GPTModule
