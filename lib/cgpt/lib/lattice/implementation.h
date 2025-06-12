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
  ViewMode view_mode;

  typedef typename Lattice<T>::vector_object vobj;
  typedef typename vobj::scalar_object sobj;
  typedef typename Lattice<T>::vector_type vCoeff_t;
  typedef typename Lattice<T>::scalar_type Coeff_t;

  cgpt_Lattice(GridBase* grid) : l(grid) {
  }

  virtual ~cgpt_Lattice() {
    //std::cout << "Deallocate" << std::endl;
  }

  cgpt_Lattice_base* create_lattice_of_same_type() {
    return new cgpt_Lattice<T>(l.Grid());
  }

  virtual std::string type() {
    return typeid(T).name();
  }

  virtual int singlet_rank() {
    return ::singlet_rank(l);
  }

  virtual PyObject* to_decl() {   
    return Py_BuildValue("(NNN)",PyLong_FromVoidPtr(this),
			PyUnicode_FromString(get_otype(l).c_str()),
			PyUnicode_FromString(get_prec(l).c_str()));
  }

  void set_to_number(ComplexD val) {
    cgpt_set_to_number(l,val);
  }

  void set_to_identity() {
    l = 1.0;
  }

  /*virtual void axpy(std::vector<cgpt_Lattice_base*>& z, std::vector<ComplexD>& a, std::vector<cgpt_Lattice_base*>& x, std::vector<cgpt_Lattice_base*>& y) {
    PVector<Lattice<T>> _x, _y, _z;
    cgpt_basis_fill(_x,x);
    cgpt_basis_fill(_y,y);
    cgpt_basis_fill(_z,z);

    cgpt_axpy(_z,a,_x,_y);
    }*/
  
  virtual void axpy(ComplexD a, cgpt_Lattice_base* x, cgpt_Lattice_base* y) {
    return cgpt_axpy(l,(Coeff_t)a,compatible<T>(x)->l,compatible<T>(y)->l);
  }
  
  virtual void scale_per_coordinate(cgpt_Lattice_base* src,ComplexD* s,int dim) {
    cgpt_scale_per_coordinate(l,compatible<T>(src)->l,s,dim);
  }

  virtual RealD norm2() {
    return ::norm2(l);
  }

  virtual void rank_inner_product(ComplexD* res, std::vector<cgpt_Lattice_base*> & left, std::vector<cgpt_Lattice_base*> & right, long n_virtual, long n_block, bool use_accelerator) {
    PVector<Lattice<T>> _left;
    PVector<Lattice<T>> _right;
    cgpt_basis_fill(_left,left);
    cgpt_basis_fill(_right,right);
    if (use_accelerator) {
      ::rankInnerProduct(res,_left,_right,n_virtual,n_block);
    } else {
      ::rankInnerProductCpu(res,_left,_right,n_virtual,n_block);
    }
  }

  virtual void inner_product_norm2(ComplexD& ip, RealD& a2, cgpt_Lattice_base* other) {
    ::innerProductNorm(ip,a2,l,compatible<T>(other)->l);
  }

  // ac == { true : add result to dst, false : replace dst }
  virtual cgpt_Lattice_base* mul(cgpt_Lattice_base* dst, bool ac, cgpt_Lattice_base* b, int unary_a, int unary_b, int unary_expr, ComplexD coef) {
    if (typeid(T) == typeid(iSinglet<vCoeff_t>) && b->type() != type()) {
      // singlet multiplication always commutes, can save half cost of instantiation
      return b->mul(dst,ac,this,unary_b,unary_a,unary_expr,coef);
    }
    return cgpt_lattice_mul(dst,ac,unary_a,l,unary_b,b,unary_expr,coef);
  }

  virtual cgpt_Lattice_base* compatible_linear_combination(cgpt_Lattice_base* dst,bool ac, std::vector<cgpt_lattice_term>& f, int unary_factor, int unary_expr) {
    return cgpt_compatible_linear_combination(l,dst,ac,f,unary_factor,unary_expr);
  }

  virtual cgpt_Lattice_base* matmul(cgpt_Lattice_base* dst, bool ac, PyArrayObject* b, std::string& bot, int unary_b, int unary_a, int unary_expr, bool reverse, ComplexD coef) {
    return cgpt_lattice_matmul(dst,ac,unary_a,l,b,bot,unary_b,unary_expr,reverse,coef);
  }

  virtual cgpt_Lattice_base* gammamul(cgpt_Lattice_base* dst, bool ac, Gamma::Algebra gamma, int unary_a, int unary_expr, bool reverse, ComplexD coef) {
    return cgpt_lattice_gammamul(dst,ac,unary_a,l,gamma,unary_expr,reverse,coef);
  }

  virtual void copy_from(cgpt_Lattice_base* _src) {
    cgpt_Lattice<T>* src = compatible<T>(_src);
    l = src->l;
  }

  virtual void fft_from(cgpt_Lattice_base* src, const std::vector<int> & dims, int sign) {
    FFT fft((GridCartesian*)l.Grid());
    Lattice<T> tmp = compatible<T>(src)->l;
    for (long i=0;i<dims.size();i++) {
      fft.FFT_dim(l,tmp,dims[i],sign);
      if (i != dims.size()-1)
	tmp = l;
    }
  }

  virtual void unary_from(cgpt_Lattice_base* a, PyObject* params) {
    cgpt_unary_from(l,compatible<T>(a)->l,params);
  }

  virtual void binary_from(cgpt_Lattice_base* a, cgpt_Lattice_base* b, PyObject* params) {
    cgpt_binary_from(l,compatible<T>(a)->l,compatible<T>(b)->l,params);
  }

  virtual void ternary_from(cgpt_Lattice_base* a, cgpt_Lattice_base* b, cgpt_Lattice_base* c, PyObject* params) {
    cgpt_ternary_from(l,
		      compatible<iSinglet<vCoeff_t>>(a)->l,
		      compatible<T>(b)->l,
		      compatible<T>(c)->l, params);
  }
  


  virtual void cshift_from(cgpt_Lattice_base* _src, int dir, int off) {
    cgpt_Lattice<T>* src = compatible<T>(_src);
    l = Cshift(src->l, dir, off);
  }

  virtual PyObject* rank_sum() {
    return cgpt_numpy_export( ::rankSum(l) );
  }

  virtual PyObject* to_str() {
    return PyUnicode_FromString(cgpt_lattice_to_str(l).c_str());
  }

  virtual void convert_from(cgpt_Lattice_base* src) {
    cgpt_lattice_convert_from(l,src);
  }

  virtual PyObject* rank_slice(std::vector<cgpt_Lattice_base*> _basis, int dim) {
    PVector<Lattice<T>> basis;
    cgpt_basis_fill(basis, _basis);
    return cgpt_lattice_rank_slice(basis, dim);
  }

  virtual PyObject* rank_indexed_sum(std::vector<cgpt_Lattice_base*> _basis, cgpt_Lattice_base* _idx, long len) {
    PVector<Lattice<T>> basis;
    cgpt_basis_fill(basis, _basis);
    return cgpt_lattice_rank_indexed_sum(basis, compatible<iSinglet<vCoeff_t>>(_idx)->l, len);
  }

  virtual void ferm_to_prop(cgpt_Lattice_base* prop, int spin, int color, bool f2p) {
    cgpt_ferm_to_prop(l,prop,spin,color,f2p);
  }

  virtual void pick_checkerboard_from(int cb, cgpt_Lattice_base* src) {
    cgpt_pickCheckerboard(cb, l, compatible<T>(src)->l);
  }

  virtual void set_checkerboard_from(cgpt_Lattice_base* src) {
    cgpt_setCheckerboard(l, compatible<T>(src)->l);
  }

  virtual void change_checkerboard(int cb) {
    l.Checkerboard() = cb;
  }

  virtual int get_checkerboard() {
    return l.Checkerboard();
  }

  virtual void basis_rotate(std::vector<cgpt_Lattice_base*> &_basis,RealD* Qt,int j0, int j1, int k0,int k1,int Nm,bool use_accelerator) {
    PVector<Lattice<T>> basis;
    cgpt_basis_fill(basis,_basis);
    cgpt_basis_rotate(basis,Qt,j0,j1,k0,k1,Nm,use_accelerator);
  }

  virtual void basis_rotate(std::vector<cgpt_Lattice_base*> &_basis,ComplexD* Qt,int j0, int j1, int k0,int k1,int Nm,bool use_accelerator) {
    PVector<Lattice<T>> basis;
    cgpt_basis_fill(basis,_basis);
    cgpt_basis_rotate(basis,Qt,j0,j1,k0,k1,Nm,use_accelerator);
  }

  virtual void linear_combination(std::vector<cgpt_Lattice_base*> & _dst, std::vector<cgpt_Lattice_base*> &_basis,ComplexD* Qt, long n_virtual, long basis_n_block) {
    PVector<Lattice<T>> basis, dst;
    cgpt_basis_fill(basis,_basis);
    cgpt_basis_fill(dst,_dst);
    cgpt_linear_combination(dst,basis,Qt,n_virtual,basis_n_block);
  }

  virtual void* memory_view_open(ViewMode mode) {
    view_mode = mode;
    if (cgpt_verbose_memory_view)
      std::cout << GridLogMessage << "cgpt::memory_view_open " << l.getHostPointer() << " mode " << mode << std::endl;
    return MemoryManager::ViewOpen(l.getHostPointer(),l.oSites()*sizeof(T), mode, l.Advise()); 
  }
  
  virtual void memory_view_close() {
    if (cgpt_verbose_memory_view)
      std::cout << GridLogMessage << "cgpt::memory_view_close " << l.getHostPointer() << " mode " << view_mode << std::endl;
    MemoryManager::ViewClose(l.getHostPointer(), view_mode);
  }

  virtual PyObject* memory_view(memory_type mt) {

    if (mt == mt_none) {
      mt = mt_host;
    }

    LatticeView<vobj>* v = new LatticeView<vobj>(l.View((mt == mt_host) ? CpuWrite : AcceleratorWrite));
    size_t sz = v->size() * sizeof((*v)[0]);
    char* ptr = (char*)&(*v)[0];

    if (cgpt_verbose_memory_view)
      std::cout << GridLogMessage << "cgpt::memory_view " << ptr << " ishost=" << (mt == mt_host) << std::endl;

    PyObject* r = PyMemoryView_FromMemory(ptr,sz,PyBUF_WRITE);
    PyObject *capsule = PyCapsule_New((void*)v, NULL, [] (PyObject *capsule) -> void { 
      LatticeView<vobj>* v = (LatticeView<vobj>*)PyCapsule_GetPointer(capsule, NULL);

      if (cgpt_verbose_memory_view) {
	char* ptr = (char*)&(*v)[0];
	std::cout << GridLogMessage << "cgpt::memory_view_close " << ptr << std::endl;
      }

      v->ViewClose();
      delete v;
    });
    ASSERT(!((PyMemoryViewObject*)r)->mbuf->master.obj);
    ((PyMemoryViewObject*)r)->mbuf->master.obj = capsule;

    return r;
  }

  virtual void describe_data_layout(long & Nsimd, long & word, long & simd_word) {
    Nsimd = l.Grid()->Nsimd();
    word = sizeof(sobj);
    simd_word = sizeof(Coeff_t);
  }

  virtual void describe_data_shape(std::vector<long> & ishape) {
    ishape.resize(0);
    cgpt_numpy_data_layout(sobj(),ishape);
    if (ishape.size() == 0) // treat complex numbers as 1d array with one element
      ishape.push_back(1);
  }

  virtual void transfer_scalar_device_buffer(std::vector<cgpt_Lattice_base*>& from, long from_n_virtual, long r, void* ptr, long size, std::vector<long>& padding, std::vector<long>& offset, bool exp, long t) {
    PVector<Lattice<T>> _from;
    cgpt_basis_fill(_from,from);
    cgpt_lattice_transfer_scalar_device_buffer(_from, from_n_virtual, r, ptr, size, padding, offset, exp, t);
  }
  
  virtual int get_numpy_dtype() {
    return infer_numpy_type(Coeff_t());
  }

  virtual cgpt_block_map_base* block_map(GridBase* coarse, 
					 std::vector<cgpt_Lattice_base*>& basis,
					 long basis_n_virtual, long basis_virtual_size, long basis_n_block,
					 cgpt_Lattice_base* mask, PyObject* tensor_projectors) {

    ASSERT(basis.size() > 0 && basis.size() % basis_n_virtual == 0);

#define BASIS_SIZE(n) if (n == basis_virtual_size) {			\
      if (tensor_projectors == Py_None) {				\
	return new cgpt_block_map<T, iVSinglet ## n<vCoeff_t>, cgpt_project_identity<T> >(coarse,basis,basis_n_virtual,basis_n_block,mask,tensor_projectors); \
      } else {								\
	return new cgpt_block_map<T, iVSinglet ## n<vCoeff_t>, cgpt_project_tensor<T> >(coarse,basis,basis_n_virtual,basis_n_block,mask,tensor_projectors); \
      }									\
    }
#include "../basis_size.h"
#undef BASIS_SIZE
    
    { ERR("Unknown basis size %d",(int)basis_virtual_size); }

  }

  virtual void invert_matrix(std::vector<cgpt_Lattice_base*>& matrix_inv, std::vector<cgpt_Lattice_base*>& matrix, long n_virtual) {
    cgpt_invert_matrix(l,matrix_inv,matrix,n_virtual);
  }

  virtual void determinant(cgpt_Lattice_base* det, std::vector<cgpt_Lattice_base*>& matrix, long n_virtual) {
    cgpt_determinant(l,det,matrix,n_virtual);
  }
  
  virtual GridBase* get_grid() {
    return l.Grid();
  }

  virtual cgpt_stencil_matrix_base* stencil_matrix(GridBase* grid, PyObject* shifts, PyObject* code, long code_parallel_block_size, long local) {
    return cgpt_stencil_matrix_create<T>(grid, shifts, code, code_parallel_block_size, local);
  }

  virtual cgpt_stencil_matrix_vector_base* stencil_matrix_vector(cgpt_Lattice_base* matrix, GridBase* grid, PyObject* shifts, PyObject* code, long code_parallel_block_size, long local,
								 int matrix_parity, int vector_parity) {
    return cgpt_stencil_matrix_vector_create<T>(matrix, grid, shifts, code, code_parallel_block_size, local, matrix_parity, vector_parity);
  }

  virtual cgpt_stencil_tensor_base* stencil_tensor(GridBase* grid, PyObject* shifts, PyObject* code, PyObject* segments, long local) {
    return cgpt_stencil_tensor_create<T>(grid, shifts, code, segments, local);
  }
};

// prevent implicit instantiation of cgpt_Lattice<>
#define INSTANTIATE(v,t,n) extern template class cgpt_Lattice<n<v>>;
#include "../instantiate/instantiate.h"
#undef INSTANTIATE
