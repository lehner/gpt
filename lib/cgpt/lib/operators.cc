/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#include "lib.h"

#include "operators/types.h"
//#include "operators/register.h"
#include "operators/base.h"
#include "operators/unary.h"
#include "operators/implementation.h"
#include "operators/wilson_clover.h"
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
