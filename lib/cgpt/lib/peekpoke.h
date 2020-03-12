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
void cgpt_lattice_poke_value(Lattice< iColourVector<vtype> >& l,std::vector<int>& coor,ComplexD& val) {
  typedef typename Lattice< iColourVector<vtype> >::vector_object vobj;
  typedef typename vobj::scalar_object sobj;
  typedef typename vtype::scalar_type Coeff_t;
  ASSERT( coor.size() == l.Grid()->Nd()+1 );
  int c1 = coor[coor.size() - 1];
  ASSERT(0 <= c1 && c1 < Nc);
  coor.resize(coor.size() - 1);
  Coordinate _coor(coor);
  iColourVector<Coeff_t> _val;
  peekSite(_val,l,_coor);
  _val._internal._internal._internal[c1] = (Coeff_t)val;
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
  ASSERT(0 <= c1 && c1 < Nc);
  ASSERT(0 <= c2 && c2 < Nc);
  coor.resize(coor.size() - 2);
  Coordinate _coor(coor);
  iColourMatrix<Coeff_t> _val;
  peekSite(_val,l,_coor);
  _val._internal._internal._internal[c1][c2] = (Coeff_t)val;
  pokeSite(_val,l,_coor);
}

template<typename vtype>
void cgpt_lattice_poke_value(Lattice< iSpinColourVector<vtype> >& l,std::vector<int>& coor,ComplexD& val) {
  typedef typename Lattice< iSpinColourVector<vtype> >::vector_object vobj;
  typedef typename vobj::scalar_object sobj;
  typedef typename vtype::scalar_type Coeff_t;
  ASSERT( coor.size() == l.Grid()->Nd()+2 );
  int s1 = coor[coor.size() - 2];
  int c1 = coor[coor.size() - 1];
  ASSERT(0 <= c1 && c1 < Nc);
  ASSERT(0 <= s1 && s1 < Ns);
  coor.resize(coor.size() - 2);
  Coordinate _coor(coor);
  iSpinColourVector<Coeff_t> _val;
  peekSite(_val,l,_coor);
  _val._internal._internal[s1]._internal[c1] = (Coeff_t)val;
  pokeSite(_val,l,_coor);
}

template<typename vtype>
void cgpt_lattice_poke_value(Lattice< iSpinColourMatrix<vtype> >& l,std::vector<int>& coor,ComplexD& val) {
  typedef typename Lattice< iSpinColourMatrix<vtype> >::vector_object vobj;
  typedef typename vobj::scalar_object sobj;
  typedef typename vtype::scalar_type Coeff_t;
  ASSERT( coor.size() == l.Grid()->Nd()+4 );
  int s1 = coor[coor.size() - 4];
  int s2 = coor[coor.size() - 3];
  int c1 = coor[coor.size() - 2];
  int c2 = coor[coor.size() - 1];
  ASSERT(0 <= c1 && c1 < Nc);
  ASSERT(0 <= c2 && c2 < Nc);
  ASSERT(0 <= s1 && s1 < Ns);
  ASSERT(0 <= s2 && s2 < Ns);
  coor.resize(coor.size() - 4);
  Coordinate _coor(coor);
  iSpinColourMatrix<Coeff_t> _val;
  peekSite(_val,l,_coor);
  _val._internal._internal[s1][s2]._internal[c1][c2] = (Coeff_t)val;
  pokeSite(_val,l,_coor);
}
