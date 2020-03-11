/*
  CGPT

  Authors: Christoph Lehner 2020
*/
class cgpt_lattice_term;
class cgpt_Lattice_base {
public:
  virtual ~cgpt_Lattice_base() { };
  virtual cgpt_Lattice_base* create_lattice_of_same_type() = 0;
  virtual void set_val(std::vector<int>& coor, ComplexD& val) = 0;
  virtual PyObject* to_str() = 0;
  virtual RealD norm2() = 0;
  virtual RealD axpy_norm(ComplexD a, cgpt_Lattice_base* x, cgpt_Lattice_base* y) = 0;
  virtual ComplexD innerProduct(cgpt_Lattice_base* other) = 0;
  virtual void copy_from(cgpt_Lattice_base* src) = 0;
  virtual cgpt_Lattice_base* mul(cgpt_Lattice_base* dst, cgpt_Lattice_base* b, int unary_a, int unary_b, int unary_expr) = 0; // unary_expr(unary_a(this) * unary_b(b))
  virtual void cshift_from(cgpt_Lattice_base* src, int dir, int off) = 0;
  virtual cgpt_Lattice_base* compatible_linear_combination(cgpt_Lattice_base* dst, std::vector<cgpt_lattice_term>& f, int unary_expr) = 0;
  virtual std::string type() = 0;
  virtual PyObject* to_decl() = 0;
};
