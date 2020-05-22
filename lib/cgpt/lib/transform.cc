/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/
#include "lib.h"

EXPORT(cshift,{
    
    void* _dst,* _src;
    PyObject* _dir,* _off;
    if (!PyArg_ParseTuple(args, "llOO", &_dst, &_src, &_dir, &_off)) {
      return NULL;
    }
    
    int dir, off;
    cgpt_Lattice_base* src = (cgpt_Lattice_base*)_src;
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
    
    cgpt_convert(_dir,dir);
    cgpt_convert(_off,off);
    
    dst->cshift_from(src,dir,off);
    
    return PyLong_FromLong(0);
  });

EXPORT(copy,{
    
    void* _dst,* _src;
    if (!PyArg_ParseTuple(args, "ll", &_dst, &_src)) {
      return NULL;
    }
    
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
    cgpt_Lattice_base* src = (cgpt_Lattice_base*)_src;
    
    dst->copy_from(src);
    
    return PyLong_FromLong(0);
  });


EXPORT(convert,{
    
    void* _dst,* _src;
    if (!PyArg_ParseTuple(args, "ll", &_dst,&_src)) {
      return NULL;
    }
    
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
    cgpt_Lattice_base* src = (cgpt_Lattice_base*)_src;
    
    dst->convert_from(src);
    
    return PyLong_FromLong(0);
  });

EXPORT(lattice_innerProduct,{
    
    void* _a,* _b;
    if (!PyArg_ParseTuple(args, "ll", &_a, &_b)) {
      return NULL;
    }
    
    cgpt_Lattice_base* a = (cgpt_Lattice_base*)_a;
    cgpt_Lattice_base* b = (cgpt_Lattice_base*)_b;
    
    ComplexD c = a->innerProduct(b);
    return PyComplex_FromDoubles(c.real(),c.imag());
  });

EXPORT(lattice_rankInnerProduct,{
    
    void* _a,* _b;
    if (!PyArg_ParseTuple(args, "ll", &_a, &_b)) {
      return NULL;
    }
    
    cgpt_Lattice_base* a = (cgpt_Lattice_base*)_a;
    cgpt_Lattice_base* b = (cgpt_Lattice_base*)_b;
    
    ComplexD c = a->rankInnerProduct(b);
    return PyComplex_FromDoubles(c.real(),c.imag());
  });

EXPORT(lattice_innerProductNorm2,{

    void* _a,* _b;
    if (!PyArg_ParseTuple(args, "ll", &_a, &_b)) {
      return NULL;
    }

    cgpt_Lattice_base* a = (cgpt_Lattice_base*)_a;
    cgpt_Lattice_base* b = (cgpt_Lattice_base*)_b;

    ComplexD ip;
    RealD a2;

    a->innerProductNorm2(ip,a2,b);

    return PyTuple_Pack(2,
                        PyComplex_FromDoubles(ip.real(),ip.imag()),
                        PyFloat_FromDouble(a2));
  });
  
EXPORT(lattice_norm2,{
    
    void* _a;
    if (!PyArg_ParseTuple(args, "l", &_a)) {
      return NULL;
    }
    
    cgpt_Lattice_base* a = (cgpt_Lattice_base*)_a;
    return PyFloat_FromDouble(a->norm2());
  });

EXPORT(lattice_axpy_norm2,{
    
    void* _r,*_x,*_y;
    PyObject* _a;
    if (!PyArg_ParseTuple(args, "lOll", &_r,&_a,&_x,&_y)) {
      return NULL;
    }
    
    cgpt_Lattice_base* x = (cgpt_Lattice_base*)_x;
    cgpt_Lattice_base* y = (cgpt_Lattice_base*)_y;
    cgpt_Lattice_base* r = (cgpt_Lattice_base*)_r;
    
    ComplexD a;
    cgpt_convert(_a,a);
    
    return PyFloat_FromDouble(r->axpy_norm2(a,x,y));
  });

EXPORT(lattice_sum,{
    
    void* _a;
    if (!PyArg_ParseTuple(args, "l", &_a)) {
      return NULL;
    }
    
    cgpt_Lattice_base* a = (cgpt_Lattice_base*)_a;
    return a->sum();
    
  });
  
EXPORT(lattice_slice,{
    
    void* _a;
    long dim;
    if (!PyArg_ParseTuple(args, "ll", &_a,&dim)) {
      return NULL;
    }
    
    cgpt_Lattice_base* a = (cgpt_Lattice_base*)_a;
    return a->slice((int)dim);
    
  });
