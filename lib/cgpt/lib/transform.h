/*
  CGPT

  Authors: Christoph Lehner 2020
*/
static PyObject* cgpt_cshift(PyObject* self, PyObject* args) {

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
}

static PyObject* cgpt_copy(PyObject* self, PyObject* args) {

  void* _dst,* _src;
  if (!PyArg_ParseTuple(args, "ll", &_dst, &_src)) {
    return NULL;
  }

  cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
  cgpt_Lattice_base* src = (cgpt_Lattice_base*)_src;

  dst->copy_from(src);

  return PyLong_FromLong(0);
}

static PyObject* cgpt_eval(PyObject* self, PyObject* args) {

  void* _dst;
  PyObject* _list;
  if (!PyArg_ParseTuple(args, "lO", &_dst, &_list)) {
    return NULL;
  }

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

  dst->eval(f,p);

  return PyLong_FromLong(0);
}

static PyObject* cgpt_lattice_mul(PyObject* self, PyObject* args) {

  void* _dst,* _a,* _b;
  if (!PyArg_ParseTuple(args, "lll", &_dst, &_a, &_b)) {
    return NULL;
  }

  cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
  cgpt_Lattice_base* a = (cgpt_Lattice_base*)_a;
  cgpt_Lattice_base* b = (cgpt_Lattice_base*)_b;

  dst->mul_from(a,b);

  return PyLong_FromLong(0);
}
