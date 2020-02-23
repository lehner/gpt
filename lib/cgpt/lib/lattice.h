/*
  CGPT

  Authors: Christoph Lehner 2020
*/
class cgpt_Lattice_base {
public:
  virtual ~cgpt_Lattice_base() { };
  virtual void set_val(std::vector<int>& coor, ComplexD& val) = 0;
  virtual PyObject* to_str() = 0;
};

template<class T>
class cgpt_Lattice : public cgpt_Lattice_base {
public:
  Lattice<T> l;

  typedef typename Lattice<T>::vector_object vobj;
  typedef typename vobj::scalar_object sobj;
  typedef typename Lattice<T>::vector_type vCoeff_t;
  typedef typename Lattice<T>::scalar_type Coeff_t;

  cgpt_Lattice(GridCartesian* grid) : l(grid) {
  }

  virtual ~cgpt_Lattice() {
    //std::cout << "Deallocate" << std::endl;
  }

  virtual void set_val(std::vector<int>& coor, ComplexD& val) {
    int nc = (int)coor.size();
    if (!nc && abs(val) == 0.0) {
      l = Zero();
    } else {
      cgpt_lattice_poke_value(l,coor,val);
    }
  }

  virtual PyObject* to_str() {
    std::stringstream st;
    st << l;
    return PyUnicode_FromString(st.str().c_str());
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
      plat = new cgpt_Lattice<vTComplexF>(grid);
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
  PyObject* _coor,* _val;
  if (!PyArg_ParseTuple(args, "lOO", &p, &_coor,&_val)) {
    return NULL;
  }

  std::vector<int> coor;
  cgpt_convert(_coor,coor);

  ComplexD val;
  cgpt_convert(_val,val);

  cgpt_Lattice_base* l = (cgpt_Lattice_base*)p;
  l->set_val(coor,val);

  return PyLong_FromLong(0);
}

static PyObject* cgpt_lattice_to_str(PyObject* self, PyObject* args) {
  void* p;
  if (!PyArg_ParseTuple(args, "l", &p)) {
    return NULL;
  }

  cgpt_Lattice_base* l = (cgpt_Lattice_base*)p;
  return l->to_str();
}
