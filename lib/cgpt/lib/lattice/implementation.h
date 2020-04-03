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

  cgpt_Lattice_base* create_lattice_of_same_type() {
    return new cgpt_Lattice<T>((GridCartesian*)l.Grid());
  }

  virtual std::string type() {
    return typeid(T).name();
  }

  virtual PyObject* to_decl() {   
    return PyTuple_Pack(3,PyLong_FromVoidPtr(this),
			PyUnicode_FromString(::get_otype(l)),
			PyUnicode_FromString(::get_prec(l))); // TODO: add l.Checkerboard()
  }

  virtual RealD axpy_norm(ComplexD a, cgpt_Lattice_base* x, cgpt_Lattice_base* y) {
    return ::axpy_norm(l,(Coeff_t)a,compatible<T>(x)->l,compatible<T>(y)->l);
  }

  virtual RealD norm2() {
    return ::norm2(l);
  }

  virtual ComplexD innerProduct(cgpt_Lattice_base* other) {
    return ::innerProduct(l,compatible<T>(other)->l);
  }

  // ac == { true : add result to dst, false : replace dst }
  virtual cgpt_Lattice_base* mul(cgpt_Lattice_base* dst, bool ac, cgpt_Lattice_base* b, int unary_a, int unary_b, int unary_expr) {
    return cgpt_lattice_mul(dst,ac,unary_a,l,unary_b,b,unary_expr);
  }

  virtual cgpt_Lattice_base* compatible_linear_combination(cgpt_Lattice_base* dst,bool ac, std::vector<cgpt_lattice_term>& f, int unary_factor, int unary_expr) {
    return cgpt_compatible_linear_combination(l,dst,ac,f,unary_factor,unary_expr);
  }

  virtual cgpt_Lattice_base* matmul(cgpt_Lattice_base* dst, bool ac, PyArrayObject* b, std::string& bot, int unary_b, int unary_a, int unary_expr, bool reverse) {
    return cgpt_lattice_matmul(dst,ac,unary_a,l,b,bot,unary_b,unary_expr,reverse);
  }

  virtual cgpt_Lattice_base* gammamul(cgpt_Lattice_base* dst, bool ac, Gamma::Algebra gamma, int unary_a, int unary_expr, bool reverse) {
    return cgpt_lattice_gammamul(dst,ac,unary_a,l,gamma,unary_expr,reverse);
  }

  virtual void copy_from(cgpt_Lattice_base* _src) {
    cgpt_Lattice<T>* src = compatible<T>(_src);
    l = src->l;
  }

  virtual void cshift_from(cgpt_Lattice_base* _src, int dir, int off) {
    cgpt_Lattice<T>* src = compatible<T>(_src);
    l = Cshift(src->l, dir, off);
  }

  virtual PyObject* get_val(const std::vector<int>& coor) {
    return cgpt_lattice_peek_value(l,coor);
  }

  virtual void set_val(const std::vector<int>& coor, PyObject* _val) {
    int nc = (int)coor.size();
    if (!nc) {
      if (cgpt_is_zero(_val)) {
	l = Zero();
      } else {
	sobj val;
	cgpt_numpy_import(val,_val);
	l = val;
      }
    } else {
      cgpt_lattice_poke_value(l,coor,_val);
    }
  }

  virtual PyObject* sum() {
    return cgpt_numpy_export( ::sum(l) );
  }

  virtual PyObject* to_str() {
    return PyUnicode_FromString(cgpt_lattice_to_str(l).c_str());
  }

  virtual void convert_from(cgpt_Lattice_base* src) {
    cgpt_lattice_convert_from(l,src);
  }

  virtual PyObject* slice(int dim) {
    return cgpt_lattice_slice(l,dim);
  }

  virtual void ferm_to_prop(cgpt_Lattice_base* prop, int spin, int color, bool f2p) {
    cgpt_ferm_to_prop(l,prop,spin,color,f2p);
  }

  virtual void pick_checkerboard_from(int cb, cgpt_Lattice_base* src) {
    pickCheckerboard(cb, l, compatible<T>(src)->l);
  }

  virtual void set_checkerboard_from(cgpt_Lattice_base* src) {
    setCheckerboard(l, compatible<T>(src)->l);
  }

  virtual void change_checkerboard(int cb) {
    l.Checkerboard() = cb;
  }

  virtual int get_checkerboard() {
    return l.Checkerboard();
  }

  virtual void basis_rotate(std::vector<cgpt_Lattice_base*> &_basis,RealD* Qt,int j0, int j1, int k0,int k1,int Nm) {
    std::vector<Lattice<T>*> basis(_basis.size());
    cgpt_basis_fill(basis,_basis);
    cgpt_basis_rotate(basis,Qt,j0,j1,k0,k1,Nm);
  }

  virtual void linear_combination(std::vector<cgpt_Lattice_base*> &_basis,RealD* Qt) {
    std::vector<Lattice<T>*> basis(_basis.size());
    cgpt_basis_fill(basis,_basis);
    cgpt_linear_combination(l,basis,Qt);
  }

  virtual PyObject* memory_view() {
    auto v = l.View();
    return PyMemoryView_FromMemory((char*)&v[0],v.size()*sizeof(v[0]),PyBUF_WRITE);
  }

  virtual PyArrayObject* export_data(PyArrayObject* coordinates) {
    return cgpt_export(l,coordinates);
  }

  virtual void import_data(PyArrayObject* coordinates, PyObject* data) {
    cgpt_import(l,coordinates,data);
  }


};

