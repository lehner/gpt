/*
  CGPT

  Authors: Christoph Lehner 2020
*/
template<typename vtype>
void cgpt_ferm_to_prop(Lattice<iSpinColourVector<vtype>>& ferm, cgpt_Lattice_base* _prop, int s, int c, bool f2p) {
  Lattice<iSpinColourMatrix<vtype>> & prop = compatible<iSpinColourMatrix<vtype>>(_prop)->l;

  if (f2p) {
    for(int j = 0; j < Ns; j++) {
      auto pjs = peekSpin(prop, j, s);
      auto fj  = peekSpin(ferm, j);
      for(int i = 0; i < Nc; i++) {
	pokeColour(pjs, peekColour(fj,i), i, c);
      }
      pokeSpin(prop, pjs, j, s);
    }
  } else {
    for(int j = 0; j < Ns; j++) {
      auto pjs = peekSpin(prop, j, s);
      auto fj  = peekSpin(ferm, j);
      for(int i = 0; i < Nc; i++) {
	pokeColour(fj, peekColour(pjs, i,c),i);
      }
      pokeSpin(ferm, fj, j);
    }
  }

}

template<typename T>
void cgpt_ferm_to_prop(Lattice<T>& ferm, cgpt_Lattice_base* _prop, int s, int c, bool f2p) {
  ERR("not supported");
}

