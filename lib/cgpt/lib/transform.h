/*
  CGPT

  Authors: Christoph Lehner 2020
*/
EXPORT_BEGIN(cshift) {

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
} EXPORT_END();

EXPORT_BEGIN(copy) {

  void* _dst,* _src;
  if (!PyArg_ParseTuple(args, "ll", &_dst, &_src)) {
    return NULL;
  }

  cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
  cgpt_Lattice_base* src = (cgpt_Lattice_base*)_src;

  dst->copy_from(src);

  return PyLong_FromLong(0);
} EXPORT_END();

EXPORT_BEGIN(eval) {

  void* _dst;
  PyObject* _list;
  int _unary;
  if (!PyArg_ParseTuple(args, "lOi", &_dst, &_list, &_unary)) {
    return NULL;
  }

  // TODO: assemble linear combinations, if there is a non-trivial field
  // operation, for now apply immediately

  cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
  assert(PyList_Check(_list));
  int n = (int)PyList_Size(_list);
  std::vector<ComplexD> f(n);
  std::vector<cgpt_Lattice_base*> p(n);

  for (int i=0;i<n;i++) {
    PyObject* tp = PyList_GetItem(_list,i);
    assert(PyTuple_Check(tp) && PyTuple_Size(tp) == 2);
    cgpt_convert(PyTuple_GetItem(tp,0),f[i]);

    PyObject* ll = PyTuple_GetItem(tp,1);
    assert(PyLong_Check(ll));
    p[i] = (cgpt_Lattice_base*)PyLong_AsVoidPtr(ll);
  }

  // for each unary pick different eval
  assert(0);
  dst->eval(f,p);

  return PyLong_FromLong(0);
} EXPORT_END();

EXPORT_BEGIN(lattice_innerProduct) {

  void* _a,* _b;
  if (!PyArg_ParseTuple(args, "ll", &_a, &_b)) {
    return NULL;
  }

  cgpt_Lattice_base* a = (cgpt_Lattice_base*)_a;
  cgpt_Lattice_base* b = (cgpt_Lattice_base*)_b;

  ComplexD c = a->innerProduct(b);
  return PyComplex_FromDoubles(c.real(),c.imag());
} EXPORT_END();

EXPORT_BEGIN(lattice_norm2) {

  void* _a;
  if (!PyArg_ParseTuple(args, "l", &_a)) {
    return NULL;
  }

  cgpt_Lattice_base* a = (cgpt_Lattice_base*)_a;
  return PyFloat_FromDouble(a->norm2());
} EXPORT_END();

EXPORT_BEGIN(lattice_axpy_norm) {

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

  return PyFloat_FromDouble(r->axpy_norm(a,x,y));
} EXPORT_END();
