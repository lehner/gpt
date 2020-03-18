/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#include "lib.h"

template<typename vtype> 
void create_lattice_prec(void* & plat, GridCartesian* grid, const vtype& t, const std::string& otype) {

#define PER_TENSOR_TYPE(T) if (otype == get_otype(T<vtype>())) {	\
    plat = new cgpt_Lattice< T< vtype > >(grid);			\
    return;								\
  } else 

#include "tensors.h"

#undef PER_TENSOR_TYPE
  { ERR("Unknown type"); }
}

EXPORT(create_lattice,{

    void* _grid;
    PyObject* _otype, * _prec;
    if (!PyArg_ParseTuple(args, "lOO", &_grid, &_otype, &_prec)) {
      return NULL;
    }
    
    GridCartesian* grid = (GridCartesian*)_grid;
    std::string otype;
    std::string prec;
    
    cgpt_convert(_otype,otype);
    cgpt_convert(_prec,prec);
    
    void* plat = 0;
    if (prec == "single") {
      vComplexF t;
      create_lattice_prec(plat,grid,t,otype);
    } else if (prec == "double") {
      vComplexD t;
      create_lattice_prec(plat,grid,t,otype);
    }
    
    if (!plat) {
      std::cerr << "Unknown field type: " << otype << "," << prec << std::endl;  
      ASSERT(0);
    }
    
    return PyLong_FromVoidPtr(plat);
  });

EXPORT(delete_lattice,{
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    delete ((cgpt_Lattice_base*)p);
    return PyLong_FromLong(0);
  });
  
EXPORT(lattice_set_val,{
    void* p;
    PyObject* _coor,* _val;
    if (!PyArg_ParseTuple(args, "lOO", &p, &_coor,&_val)) {
      return NULL;
    }
    
    std::vector<int> coor;
    cgpt_convert(_coor,coor);
    
    cgpt_Lattice_base* l = (cgpt_Lattice_base*)p;
    l->set_val(coor,_val);
    
    return PyLong_FromLong(0);
  });

EXPORT(lattice_get_val,{
    void* p;
    PyObject* _coor;
    if (!PyArg_ParseTuple(args, "lO", &p, &_coor)) {
      return NULL;
    }
    
    std::vector<int> coor;
    cgpt_convert(_coor,coor);
    
    cgpt_Lattice_base* l = (cgpt_Lattice_base*)p;
    return l->get_val(coor);
    
  });

EXPORT(lattice_to_str,{
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    cgpt_Lattice_base* l = (cgpt_Lattice_base*)p;
    return l->to_str();
  });

EXPORT(lattice_pick_checkerboard,{
    void* _src, *_dst;
    long cb;
    if (!PyArg_ParseTuple(args, "lOO", &cb, &_src,&_dst)) {
      return NULL;
    }
    cgpt_Lattice_base* src = (cgpt_Lattice_base*)_src;
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
    dst->pick_checkerboard_from(cb == 0 ? Even : Odd, src);
    
    return PyLong_FromLong(0);
  });

EXPORT(lattice_set_checkerboard,{
    void* _src, *_dst;
    if (!PyArg_ParseTuple(args, "OO", &_src,&_dst)) {
      return NULL;
    }
    cgpt_Lattice_base* src = (cgpt_Lattice_base*)_src;
    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
    dst->set_checkerboard_from(src);
    
    return PyLong_FromLong(0);
  });
