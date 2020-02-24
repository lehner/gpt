/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#define typeis(a,at) ( a->type() == typeid(at).name() )
#define castas(a,at) ( ((cgpt_Lattice<at>*)a)->l )
#define LAT_MUL_IMPL(at,bt) if (typeis(a,at) && typeis(b,bt)) { l = castas(a,at) * castas(b,bt); return; }

void cgpt_lattice_mul_from(Lattice<vTComplexF>& l,cgpt_Lattice_base* a,cgpt_Lattice_base* b) {
  LAT_MUL_IMPL(vTComplexF,vTComplexF);
  std::cerr << "Not implemented" << std::endl;
  assert(0);
}

void cgpt_lattice_mul_from(Lattice<vTComplexD>& l,cgpt_Lattice_base* a,cgpt_Lattice_base* b) {
  LAT_MUL_IMPL(vTComplexD,vTComplexD);
  std::cerr << "Not implemented" << std::endl;
  assert(0);
}
