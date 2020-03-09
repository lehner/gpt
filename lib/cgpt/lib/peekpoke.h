/*
  CGPT

  Authors: Christoph Lehner 2020
*/
template<typename vtype>
void cgpt_lattice_poke_value(Lattice< iSinglet<vtype> >& l,std::vector<int>& coor,ComplexD& val) {
  typedef typename vtype::scalar_type Coeff_t;
  ASSERT( coor.size() == l.Grid()->Nd() );
  Coordinate _coor(coor);
  iSinglet<Coeff_t> _val = { (Coeff_t)val };
  pokeSite(_val,l,_coor);
}

template<typename vtype>
void cgpt_lattice_poke_value(Lattice< iColourMatrix<vtype> >& l,std::vector<int>& coor,ComplexD& val) {
  typedef typename Lattice< iColourMatrix<vtype> >::vector_object vobj;
  typedef typename vobj::scalar_object sobj;
  typedef typename vtype::scalar_type Coeff_t;
  ASSERT( coor.size() == l.Grid()->Nd()+2 );
  int c1 = coor[coor.size() - 2];
  int c2 = coor[coor.size() - 1];
  ASSERT(0 <= c1 && c1 < 3);
  ASSERT(0 <= c2 && c2 < 3);
  coor.resize(coor.size() - 2);
  Coordinate _coor(coor);
  iColourMatrix<Coeff_t> _val;
  peekSite(_val,l,_coor);
  _val._internal._internal._internal[c1][c2] = (Coeff_t)val;
  pokeSite(_val,l,_coor);
}
