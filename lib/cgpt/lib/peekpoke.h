/*
  CGPT

  Authors: Christoph Lehner 2020
*/
void cgpt_lattice_poke_value(Lattice<vTComplexF>& l,std::vector<int>& coor,ComplexD& val) {
  assert( coor.size() == l.Grid()->Nd() );
  Coordinate _coor(coor);
  TComplexF _val = { (ComplexF)val };
  pokeSite(_val,l,_coor);
}
