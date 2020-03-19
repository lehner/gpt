/*
  CGPT

  Authors: Christoph Lehner 2020
*/
template<typename T>
class cgpt_fermion_operator : public cgpt_fermion_operator_base {
public:
  T* op;

  cgpt_fermion_operator(T* _op) : op(_op) {
  }

  virtual ~cgpt_fermion_operator() { 
    delete op;
  }

  virtual RealD unary(int opcode, cgpt_Lattice_base* in, cgpt_Lattice_base* out) {
    return cgpt_fermion_operator_unary<T>(*op,opcode,in,out);
  }

};
