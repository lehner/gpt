/*
  CGPT

  Authors: Christoph Lehner 2020
*/
class cgpt_fermion_operator_base {
public:
  virtual ~cgpt_fermion_operator_base() { };
  virtual RealD unary(int opcode, cgpt_Lattice_base* in, cgpt_Lattice_base* out) = 0;
};
