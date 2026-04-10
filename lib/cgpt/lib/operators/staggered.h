/*
    GPT - Grid Python Toolkit
    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/
template<typename vCoeff_t>
cgpt_fermion_operator_base* cgpt_create_staggered(PyObject* args) {

  typedef StaggeredImpl<vCoeff_t, FundamentalRepresentation > SI;
  typedef typename vCoeff_t::scalar_type Coeff_t;
  typename SI::ImplParams sp;

  auto grid = get_pointer<GridCartesian>(args,"U_grid");
  auto grid_rb = get_pointer<GridRedBlackCartesian>(args,"U_grid_rb");
  RealD mass = get_float(args,"mass");
  RealD c1 = get_float(args,"c1");
  RealD c2 = get_float(args,"c2");
  RealD u0 = get_float(args,"u0");
  // sp.boundary_phases = get_complex_vec<Nd>(args,"boundary_phases");  <--- not yet implemented in Grid

  Lattice< iLorentzColourMatrix< vCoeff_t > > U(grid), U2(grid);
  for (int mu=0;mu<Nd;mu++) {
    PokeIndex<LorentzIndex>(U,compatible<iColourMatrix<vCoeff_t>>(get_pointer<cgpt_Lattice_base>(args,"U",mu))->l,mu);
    PokeIndex<LorentzIndex>(U2,compatible<iColourMatrix<vCoeff_t>>(get_pointer<cgpt_Lattice_base>(args,"U",mu+Nd))->l,mu);
  }

  auto f = new ImprovedStaggeredFermion<SI>(U,U2,*grid,*grid_rb,mass,c1,c2,u0,sp);
  return new cgpt_fermion_operator_staggered<ImprovedStaggeredFermion<SI>>(f);
}
