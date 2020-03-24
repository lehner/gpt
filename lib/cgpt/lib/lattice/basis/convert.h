/*
  CGPT

  Authors: Christoph Lehner 2020
*/

template<typename T>
void cgpt_basis_fill(std::vector<Lattice<T>*>& basis, std::vector<cgpt_Lattice_base*>& _basis) {
  basis.resize(_basis.size());
  for (size_t i=0;i<basis.size();i++)
    basis[i] = &compatible<T>(_basis[i])->l;
}

static void cgpt_basis_fill(std::vector<cgpt_Lattice_base*>& basis, PyObject* _basis) {
  ASSERT(PyList_Check(_basis));
  Py_ssize_t size = PyList_Size(_basis);
  basis.resize(size);
  for (Py_ssize_t i=0;i<size;i++) {
    PyObject* li = PyList_GetItem(_basis,i);
    PyObject* obj = PyObject_GetAttrString(li,"obj");
    ASSERT(obj);
    ASSERT(PyLong_Check(obj));
    basis[i] = (cgpt_Lattice_base*)PyLong_AsVoidPtr(obj);
  }
}
