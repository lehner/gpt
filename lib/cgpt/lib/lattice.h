/*
  CGPT

  Authors: Christoph Lehner 2020
*/
class cgpt_Lattice_base {
public:
  virtual ~cgpt_Lattice_base() { };
};

template<class t>
class cgpt_Lattice : public cgpt_Lattice_base {
public:
  Lattice<t> l;

  cgpt_Lattice(GridCartesian* grid) : l(grid) {
  }

  virtual ~cgpt_Lattice() {
    std::cout << "Deallocate" << std::endl;
  }
};

static PyObject* cgpt_create_lattice(PyObject* self, PyObject* args) {

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
  if (otype == "complex") {
    if (prec == "single") {
      plat = new cgpt_Lattice<vComplexF>(grid);
    }
  }

  if (!plat) {
    std::cerr << "Unknown field type: " << otype << "," << prec << std::endl;  
    assert(0);
  }

  return PyLong_FromVoidPtr(plat);
}


static PyObject* cgpt_delete_lattice(PyObject* self, PyObject* args) {
  void* p;
  if (!PyArg_ParseTuple(args, "l", &p)) {
    return NULL;
  }

  delete ((cgpt_Lattice_base*)p);
  return PyLong_FromLong(0);
}

static PyObject* cgpt_lattice_set_val(PyObject* self, PyObject* args) {
  void* p;
  if (!PyArg_ParseTuple(args, "l", &p)) {
    return NULL;
  }

  cgpt_Lattice_base* l = (cgpt_Lattice_base*)p;
  return PyLong_FromLong(0);
}

static PyObject* cgpt_lattice_to_dict(PyObject* self, PyObject* args) {
  void* p;
  if (!PyArg_ParseTuple(args, "l", &p)) {
    return NULL;
  }

  cgpt_Lattice_base* l = (cgpt_Lattice_base*)p;
  return PyLong_FromLong(0);
}
