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
class infrequent_use:
    tag="infrequent_use"

class to_host:
    tag="host"

class to_accelerator:
    tag="accelerator"

def distribute(o,f):
    if type(o) == list:
        return [ distribute(i,f) for i in o ]
    elif type(o) == tuple:
        return tuple(distribute(list(o),f))
    elif type(o) == dict:
        return { i : distribute(o[i],f) for i in o }
    else:
        return f(o)

def advise(o,t):
    return distribute(o, lambda x: x.advise(t))

def prefetch(o,t):
    return distribute(o, lambda x: x.prefetch(t))

