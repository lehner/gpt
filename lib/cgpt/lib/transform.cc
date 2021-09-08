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

EXPORT(fft,{
    
    void* _dst,* _src;
    PyObject* _dims, *_sign;
    if (!PyArg_ParseTuple(args, "llOO", &_dst, &_src, &_dims, &_sign)) {
      return NULL;
    }
    
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
    cgpt_Lattice_base* src = (cgpt_Lattice_base*)_src;

    std::vector<int> dims;
    int sign;
    cgpt_convert(_dims,dims);
    cgpt_convert(_sign,sign);
    
    dst->fft_from(src,dims,sign);
    
    return PyLong_FromLong(0);
  });

EXPORT(unary,{
    
    void* _dst,* _src;
    PyObject* params;
    if (!PyArg_ParseTuple(args, "llO", &_dst, &_src, &params)) {
      return NULL;
    }
    
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
    cgpt_Lattice_base* src = (cgpt_Lattice_base*)_src;
    
    dst->unary_from(src,params);
    
    return PyLong_FromLong(0);
  });

EXPORT(binary,{
    
    void* _dst,* _a, * _b;
    PyObject* params;
    if (!PyArg_ParseTuple(args, "lllO", &_dst, &_a, &_b, &params)) {
      return NULL;
    }
    
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
    cgpt_Lattice_base* a = (cgpt_Lattice_base*)_a;
    cgpt_Lattice_base* b = (cgpt_Lattice_base*)_b;
    
    dst->binary_from(a,b,params);
    
    return PyLong_FromLong(0);
  });

EXPORT(ternary,{
    
    void* _dst,* _a, * _b, * _c;
    PyObject* params;
    if (!PyArg_ParseTuple(args, "llllO", &_dst, &_a, &_b, &_c, &params)) {
      return NULL;
    }
    
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
    cgpt_Lattice_base* a = (cgpt_Lattice_base*)_a;
    cgpt_Lattice_base* b = (cgpt_Lattice_base*)_b;
    cgpt_Lattice_base* c = (cgpt_Lattice_base*)_c;
    
    dst->ternary_from(a,b,c,params);
    
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

EXPORT(lattice_rank_inner_product,{
    
    PyObject* _left,* _right;
    long use_accelerator;
    if (!PyArg_ParseTuple(args, "OOl", &_left, &_right, &use_accelerator)) {
      return NULL;
    }
    
    std::vector<cgpt_Lattice_base*> left, right;
    long n_virtual_left = cgpt_basis_fill(left,_left);
    long n_virtual_right = cgpt_basis_fill(right,_right);
    ASSERT(n_virtual_left == n_virtual_right);

    std::vector<long> dim(2);
    dim[0] = left.size() / n_virtual_left;
    dim[1] = right.size() / n_virtual_right;

    PyArrayObject* ret = cgpt_new_PyArray((int)dim.size(), &dim[0], NPY_COMPLEX128);
    ComplexD* result = (ComplexD*)PyArray_DATA(ret);

    ASSERT(left.size() > 0);
    
    left[0]->rank_inner_product(result,left,right,n_virtual_left,use_accelerator);

    return (PyObject*)ret;
  });

EXPORT(lattice_inner_product_norm2,{

    void* _a,* _b;
    if (!PyArg_ParseTuple(args, "ll", &_a, &_b)) {
      return NULL;
    }

    cgpt_Lattice_base* a = (cgpt_Lattice_base*)_a;
    cgpt_Lattice_base* b = (cgpt_Lattice_base*)_b;

    ComplexD ip;
    RealD a2;

    a->inner_product_norm2(ip,a2,b);

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

EXPORT(lattice_axpy,{
    
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

    r->axpy(a,x,y);
    
    return PyLong_FromLong(0);
  });

EXPORT(lattice_scale_per_coordinate,{
    
    void* _d,*_s;
    PyObject* _a;
    long dim;
    if (!PyArg_ParseTuple(args, "llOl", &_d,&_s,&_a,&dim)) {
      return NULL;
    }
    
    cgpt_Lattice_base* d = (cgpt_Lattice_base*)_d;
    cgpt_Lattice_base* s = (cgpt_Lattice_base*)_s;

    ComplexD* a;
    int L;
    cgpt_numpy_import_vector(_a,a,L);

    ASSERT((0 <= dim) && (dim < d->get_grid()->_gdimensions.size()));
    ASSERT(d->get_grid()->_gdimensions[dim] == L);

    d->scale_per_coordinate(s,a,dim);
    
    return PyLong_FromLong(0);
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
    
    PyObject* _basis;
    long dim;
    if (!PyArg_ParseTuple(args, "Ol", &_basis, &dim)) {
      return NULL;
    }
    
    std::vector<cgpt_Lattice_base*> basis;
    cgpt_basis_fill(basis,_basis);

    return basis[0]->slice(basis, (int)dim);
    
  });
