/*
  CGPT

  Authors: Christoph Lehner 2020
*/
class cgpt_Lattice_base {
public:
  virtual ~cgpt_Lattice_base() { };
  virtual void set_val(std::vector<int>& coor, ComplexD& val) = 0;
  virtual PyObject* to_str() = 0;
  virtual RealD norm2() = 0;
  virtual RealD axpy_norm(ComplexD a, cgpt_Lattice_base* x, cgpt_Lattice_base* y) = 0;
  virtual ComplexD innerProduct(cgpt_Lattice_base* other) = 0;
  virtual void adj_from(cgpt_Lattice_base* other) = 0;
  virtual void copy_from(cgpt_Lattice_base* src) = 0;
  virtual void mul_from(cgpt_Lattice_base* a, cgpt_Lattice_base* b) = 0;
  virtual void cshift_from(cgpt_Lattice_base* src, int dir, int off) = 0;
  virtual void eval(std::vector<ComplexD>& f, std::vector<cgpt_Lattice_base*>& l) = 0;
  virtual std::string type() = 0;
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

  virtual std::string type() {
    return typeid(T).name();
  }

  virtual cgpt_Lattice<T>* compatible(cgpt_Lattice_base* other) {
    assert(type() == other->type());
    return (cgpt_Lattice<T>*)other;
  }

  virtual RealD axpy_norm(ComplexD a, cgpt_Lattice_base* x, cgpt_Lattice_base* y) {
    return ::axpy_norm(l,(Coeff_t)a,compatible(x)->l,compatible(y)->l);
  }

  virtual RealD norm2() {
    return ::norm2(l);
  }

  virtual ComplexD innerProduct(cgpt_Lattice_base* other) {
    return ::innerProduct(l,compatible(other)->l);
  }

  virtual void adj_from(cgpt_Lattice_base* other) {
    l = adj(compatible(other)->l);
  }

  virtual void mul_from(cgpt_Lattice_base* a, cgpt_Lattice_base* b) {
    (void)cgpt_lattice_mul_from(l,a,b);
  }

  // In current model need eval, evalTrace, evalSpinTrace, evalColorTrace; maybe do this with templates?
  virtual void eval(std::vector<ComplexD>& f, std::vector<cgpt_Lattice_base*>& a) {
    int n = (int)f.size();
    assert(f.size() == a.size());
#define EF(i) ((Coeff_t)f[i]) * compatible(a[i])->l
    if (n == 0) {
      l = Zero();
    } else if (n == 1) {
      l = EF(0);
    } else if (n == 2) {
      l = EF(0) + EF(1);
    } else if (n == 3) {
      l = EF(0) + EF(1) + EF(2);
    } else if (n == 4) {
      l = EF(0) + EF(1) + EF(2) + EF(3);
    } else if (n == 5) {
      l = EF(0) + EF(1) + EF(2) + EF(3) + EF(4);
    } else if (n == 6) {
      l = EF(0) + EF(1) + EF(2) + EF(3) + EF(4) + EF(5);
    } else if (n == 7) {
      l = EF(0) + EF(1) + EF(2) + EF(3) + EF(4) + EF(5) + EF(6);
    } else if (n == 8) {
      l = EF(0) + EF(1) + EF(2) + EF(3) + EF(4) + EF(5) + EF(6) + EF(7);
    } else if (n == 9) {
      l = EF(0) + EF(1) + EF(2) + EF(3) + EF(4) + EF(5) + EF(6) + EF(7) + EF(8);
    } else {
      std::cerr << "Need to hard-code linear combination with n = " << n << std::endl;
      assert(0);
    }
  }

  virtual void copy_from(cgpt_Lattice_base* _src) {
    cgpt_Lattice<T>* src = compatible(_src);
    l = src->l;
  }

  virtual void cshift_from(cgpt_Lattice_base* _src, int dir, int off) {
    cgpt_Lattice<T>* src = compatible(_src);
    l = Cshift(src->l, dir, off);
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

EXPORT_BEGIN(create_lattice) {

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
  if (otype == "ot_complex") {
    if (prec == "single") {
      plat = new cgpt_Lattice<vTComplexF>(grid);
    } else if (prec == "double") {
      plat = new cgpt_Lattice<vTComplexD>(grid);
    }
  } else if (otype == "ot_mcolor") {
    if (prec == "single") {
      plat = new cgpt_Lattice<vColourMatrixF>(grid);
    } else if (prec == "double") {
      plat = new cgpt_Lattice<vColourMatrixD>(grid);
    }
  }


  if (!plat) {
    std::cerr << "Unknown field type: " << otype << "," << prec << std::endl;  
    assert(0);
  }

  return PyLong_FromVoidPtr(plat);
} EXPORT_END();


EXPORT_BEGIN(delete_lattice) {
  void* p;
  if (!PyArg_ParseTuple(args, "l", &p)) {
    return NULL;
  }

  delete ((cgpt_Lattice_base*)p);
  return PyLong_FromLong(0);
} EXPORT_END();

EXPORT_BEGIN(lattice_set_val) {
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
} EXPORT_END();

EXPORT_BEGIN(lattice_to_str) {
  void* p;
  if (!PyArg_ParseTuple(args, "l", &p)) {
    return NULL;
  }

  cgpt_Lattice_base* l = (cgpt_Lattice_base*)p;
  return l->to_str();
} EXPORT_END();
