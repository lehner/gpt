/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#define typeis(a,at) ( a->type() == typeid(at).name() )
#define castas(a,at) ( ((cgpt_Lattice<at>*)a)->l )
#define LAT_MUL_IMPL(at,bt) if (typeis(a,at) && typeis(b,bt)) { l = castas(a,at) * castas(b,bt); return; }

template<typename vtype>
void cgpt_lattice_mul_from(Lattice< iSinglet<vtype> >& l,cgpt_Lattice_base* a,cgpt_Lattice_base* b) {
  LAT_MUL_IMPL(iSinglet<vtype>,iSinglet<vtype>);
  std::cerr << "Not implemented" << std::endl;
  ASSERT(0);
}

template<typename vtype>
void cgpt_lattice_mul_from(Lattice< iColourMatrix<vtype> >& l,cgpt_Lattice_base* a,cgpt_Lattice_base* b) {
  LAT_MUL_IMPL(iColourMatrix<vtype>,iColourMatrix<vtype>);
  std::cerr << "Not implemented" << std::endl;
  ASSERT(0);
}
