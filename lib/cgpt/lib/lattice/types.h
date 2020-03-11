/*
  CGPT

  Authors: Christoph Lehner 2020
*/
const char* get_prec(const vComplexF& l) { return "single"; };
const char* get_prec(const vComplexD& l) { return "double"; };
template<typename T> const char* get_prec(const Lattice<T>& l) { typedef typename Lattice<T>::vector_type vCoeff_t; vCoeff_t t; return get_prec(t); }
template<typename vobj> const char* get_otype(const Lattice<iSinglet<vobj>>& l) { return "ot_complex"; };
template<typename vobj> const char* get_otype(const Lattice<iColourMatrix<vobj>>& l) { return "ot_mcolor"; };

