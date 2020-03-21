/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#define UNOP_VOID(name,opcode)						\
  case opcode: op.name(compatible<vobj>(in)->l,compatible<vobj>(out)->l); return 0.0;

#define UNOP_REALD(name,opcode)						\
  case opcode: return op.name(compatible<vobj>(in)->l,compatible<vobj>(out)->l);

template<typename T>
RealD cgpt_fermion_operator_unary(T& op, int opcode, cgpt_Lattice_base* in,cgpt_Lattice_base* out) {
  typedef typename T::FermionField::vector_object vobj;
  
  switch (opcode) {
#include "register.h"
  default:
    ERR("Unknown opcode %d",opcode);
  }
}

#undef UNOP
