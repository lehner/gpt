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
class cgpt_block_map_base;
class cgpt_lattice_term;
class cgpt_Lattice_base {
public:
  virtual ~cgpt_Lattice_base() { };
  virtual cgpt_Lattice_base* create_lattice_of_same_type() = 0;
  virtual void set_to_number(ComplexD val) = 0;
  virtual PyObject* to_str() = 0;
  virtual PyObject* sum() = 0;
  virtual RealD norm2() = 0;
  virtual void axpy(ComplexD a, cgpt_Lattice_base* x, cgpt_Lattice_base* y) = 0;
  virtual void rank_inner_product(ComplexD* result, std::vector<cgpt_Lattice_base*> & left, std::vector<cgpt_Lattice_base*> & right, long n_virtual, bool use_accelerator) = 0;
  virtual void inner_product_norm2(ComplexD& ip, RealD& a2, cgpt_Lattice_base* other) = 0;
  virtual void copy_from(cgpt_Lattice_base* src) = 0;
  virtual void fft_from(cgpt_Lattice_base* src, const std::vector<int> & dims, int sign) = 0;
  virtual void unary_from(cgpt_Lattice_base* src, PyObject* params) = 0;
  virtual void binary_from(cgpt_Lattice_base* a, cgpt_Lattice_base* b, PyObject* params) = 0;
  virtual void ternary_from(cgpt_Lattice_base* question, cgpt_Lattice_base* yes, cgpt_Lattice_base* no, PyObject* params) = 0;
  virtual cgpt_Lattice_base* mul(cgpt_Lattice_base* dst, bool ac, cgpt_Lattice_base* b, int unary_a, int unary_b, int unary_expr, ComplexD coef) = 0; // unary_expr(unary_a(this) * unary_b(b)) * coef
  virtual cgpt_Lattice_base* matmul(cgpt_Lattice_base* dst, bool ac, PyArrayObject* b, std::string& bot, int unary_b, int unary_a, int unary_expr, bool reverse, ComplexD coef) = 0;
  virtual cgpt_Lattice_base* gammamul(cgpt_Lattice_base* dst, bool ac, Gamma::Algebra gamma, int unary_a, int unary_expr, bool reverse, ComplexD coef) = 0;
  virtual void cshift_from(cgpt_Lattice_base* src, int dir, int off) = 0;
  virtual cgpt_Lattice_base* compatible_linear_combination(cgpt_Lattice_base* dst, bool ac, std::vector<cgpt_lattice_term>& f, int unary_factor, int unary_expr) = 0;
  virtual std::string type() = 0;
  virtual int singlet_rank() = 0;
  virtual PyObject* to_decl() = 0;
  virtual void convert_from(cgpt_Lattice_base* src) = 0;
  virtual PyObject* slice(std::vector<cgpt_Lattice_base*> _basis, int dim) = 0;
  virtual void ferm_to_prop(cgpt_Lattice_base* prop, int spin, int color, bool f2p) = 0;
  virtual void pick_checkerboard_from(int cb, cgpt_Lattice_base* src) = 0;
  virtual void set_checkerboard_from(cgpt_Lattice_base* src) = 0;
  virtual void change_checkerboard(int cb) = 0;
  virtual int get_checkerboard() = 0;
  virtual void scale_per_coordinate(cgpt_Lattice_base* src,ComplexD* s,int dim) = 0;
  virtual void basis_rotate(std::vector<cgpt_Lattice_base*> &basis,RealD* Qt,int j0, int j1, int k0,int k1,int Nm,bool use_accelerator) = 0;
  virtual void basis_rotate(std::vector<cgpt_Lattice_base*> &basis,ComplexD* Qt,int j0, int j1, int k0,int k1,int Nm,bool use_accelerator) = 0;
  virtual void linear_combination(std::vector<cgpt_Lattice_base*>& dst, std::vector<cgpt_Lattice_base*> &basis,ComplexD* Qt, long n_virtual, long basis_n_block) = 0;
  virtual PyObject* memory_view(memory_type mt) = 0; // access to internal memory storage, can be simd format
  virtual void describe_data_layout(long & Nsimd, long & word, long & simd_word, std::vector<long> & ishape) = 0;
  virtual int get_numpy_dtype() = 0;
  virtual cgpt_block_map_base* block_map(GridBase* coarse, std::vector<cgpt_Lattice_base*>& basis, 
					 long basis_n_virtual, long basis_virtual_size, long basis_n_block,
					 cgpt_Lattice_base* mask) = 0;
  virtual void invert_matrix(std::vector<cgpt_Lattice_base*>& matrix_inv, std::vector<cgpt_Lattice_base*>& matrix, long n_virtual) = 0;
  virtual void determinant(cgpt_Lattice_base* det, std::vector<cgpt_Lattice_base*>& matrix, long n_virtual) = 0; // this determines type of matrix[0]
  virtual GridBase* get_grid() = 0;
};

template<class T> class cgpt_Lattice;

template<typename T>
cgpt_Lattice<T>* compatible(cgpt_Lattice_base* other) {
  if (typeid(T).name() != other->type()) {
    std::string expected_name = demangle(typeid(T).name());
    std::string given_name = demangle(other->type().c_str());
    ERR("Expected type %s, got type %s", expected_name.c_str(), given_name.c_str());
  }
  return (cgpt_Lattice<T>*)other;
}
