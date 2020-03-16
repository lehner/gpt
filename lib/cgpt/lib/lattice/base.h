/*
  CGPT

  Authors: Christoph Lehner 2020
*/
class cgpt_lattice_term;
class cgpt_Lattice_base {
public:
  virtual ~cgpt_Lattice_base() { };
  virtual cgpt_Lattice_base* create_lattice_of_same_type() = 0;
  virtual void set_val(const std::vector<int>& coor, PyObject* val) = 0;
  virtual PyObject* get_val(const std::vector<int>& coor) = 0;
  virtual PyObject* to_str() = 0;
  virtual PyObject* sum() = 0;
  virtual RealD norm2() = 0;
  virtual RealD axpy_norm(ComplexD a, cgpt_Lattice_base* x, cgpt_Lattice_base* y) = 0;
  virtual ComplexD innerProduct(cgpt_Lattice_base* other) = 0;
  virtual void copy_from(cgpt_Lattice_base* src) = 0;
  virtual cgpt_Lattice_base* mul(cgpt_Lattice_base* dst, bool ac, cgpt_Lattice_base* b, int unary_a, int unary_b, int unary_expr) = 0; // unary_expr(unary_a(this) * unary_b(b))
  virtual cgpt_Lattice_base* matmul(cgpt_Lattice_base* dst, bool ac, PyArrayObject* b, std::string& bot, int unary_b, int unary_a, int unary_expr, bool reverse) = 0;
  virtual cgpt_Lattice_base* gammamul(cgpt_Lattice_base* dst, bool ac, int gamma, int unary_a, int unary_expr, bool reverse) = 0;
  virtual void cshift_from(cgpt_Lattice_base* src, int dir, int off) = 0;
  virtual cgpt_Lattice_base* compatible_linear_combination(cgpt_Lattice_base* dst, bool ac, std::vector<cgpt_lattice_term>& f, int unary_factor, int unary_expr) = 0;
  virtual std::string type() = 0;
  virtual PyObject* to_decl() = 0;
  virtual void convert_from(cgpt_Lattice_base* dst) = 0;
  virtual PyObject* slice(int dim) = 0;
};
