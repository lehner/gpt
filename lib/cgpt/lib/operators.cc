/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
                  2020  Daniel Richtmann (daniel.richtmann@ur.de)

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

#include "operators/types.h"
//#include "operators/register.h"
#include "operators/base.h"
#include "operators/unary.h"
#include "operators/implementation.h"
#include "operators/wilson_clover.h"
#include "operators/zmobius.h"
#include "operators/mobius.h"
#include "operators/coarse.h"
#include "operators/create.h"

EXPORT(create_fermion_operator,{

    PyObject* _optype,* _args,* _prec;
    if (!PyArg_ParseTuple(args, "OOO", &_optype, &_prec, &_args)) {
      return NULL;
    }

    std::string optype, prec;
    cgpt_convert(_optype,optype);
    cgpt_convert(_prec,prec);

    void* pop = 0;
    if (prec == "single") {
      pop = cgpt_create_fermion_operator<vComplexF>(optype,_args);
    } else if (prec == "double") {
      pop = cgpt_create_fermion_operator<vComplexD>(optype,_args);
    } else {
      ERR("Unknown precision");
    }

    ASSERT(pop);

    return PyLong_FromVoidPtr(pop);
  });

EXPORT(update_fermion_operator,{
    
    void* p;
    PyObject* _args;
    if (!PyArg_ParseTuple(args, "lO", &p, &_args)) {
      return NULL;
    }
    
    ((cgpt_fermion_operator_base*)p)->update(_args);

    return PyLong_FromLong(0);
  });

EXPORT(delete_fermion_operator,{

    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }

    delete ((cgpt_fermion_operator_base*)p);
    return PyLong_FromLong(0);
  });

EXPORT(apply_fermion_operator,{

    void* p, *_src, *_dst;
    long op;
    if (!PyArg_ParseTuple(args, "llll", &p,&op,&_src,&_dst)) {
      return NULL;
    }

    cgpt_Lattice_base* src = (cgpt_Lattice_base*)_src;
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;

    return PyFloat_FromDouble( ((cgpt_fermion_operator_base*)p)->unary((int)op,src,dst) );

  });

EXPORT(apply_fermion_deriv,{

    void* p, *_f0, *_f1, *_f2, *_f3, *_X, *_Y;
    long dag;
    if (!PyArg_ParseTuple(args, "l(llll)lll", &p,&_f0,&_f1,&_f2,&_f3,&_X,&_Y,&dag)) {
      return NULL;
    }

    cgpt_Lattice_base* f0 = (cgpt_Lattice_base*)_f0;
    cgpt_Lattice_base* f1 = (cgpt_Lattice_base*)_f1;
    cgpt_Lattice_base* f2 = (cgpt_Lattice_base*)_f2;
    cgpt_Lattice_base* f3 = (cgpt_Lattice_base*)_f3;
    cgpt_Lattice_base* X = (cgpt_Lattice_base*)_X;
    cgpt_Lattice_base* Y = (cgpt_Lattice_base*)_Y;

    ((cgpt_fermion_operator_base*)p)->deriv({f0,f1,f2,f3}, X, Y, dag);
    return PyLong_FromLong(0);

  });

EXPORT(apply_fermion_operator_dirdisp,{

    void* p, *_src, *_dst;
    long op, dir, disp;
    if (!PyArg_ParseTuple(args, "llllll", &p,&op,&_src,&_dst,&dir,&disp)) {
      return NULL;
    }

    cgpt_Lattice_base* src = (cgpt_Lattice_base*)_src;
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;

    return PyFloat_FromDouble( ((cgpt_fermion_operator_base*)p)->dirdisp((int)op,src,dst,(int)dir,(int)disp) );

  });
