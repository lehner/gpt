/*
  CGPT

  Authors: Christoph Lehner 2020
*/
template<typename vCoeff_t>
cgpt_fermion_operator_base* cgpt_create_zmobius(PyObject* args) {

  typedef WilsonImpl<vCoeff_t, FundamentalRepresentation, CoeffComplex > WI;
  typedef typename vCoeff_t::scalar_type Coeff_t;

  typename WI::ImplParams wp;

  auto grid5 = get_pointer<GridCartesian>(args,"F_grid");
  auto grid5_rb = get_pointer<GridRedBlackCartesian>(args,"F_grid_rb");
  auto grid4 = get_pointer<GridCartesian>(args,"U_grid");
  auto grid4_rb = get_pointer<GridRedBlackCartesian>(args,"U_grid_rb");
  RealD mass = get_float(args,"mass");
  RealD M5 = get_float(args,"M5");
  RealD b = get_float(args,"b");
  RealD c = get_float(args,"c");
  std::vector<ComplexD> omega = get_complex_vec_gen(args,"omega");
  wp.boundary_phases = get_complex_vec<Nd>(args,"boundary_phases");
  wp.overlapCommsCompute = true;

  Lattice< iLorentzColourMatrix< vCoeff_t > > U(grid4);
  for (int mu=0;mu<Nd;mu++) {
    auto l = get_pointer<cgpt_Lattice_base>(args,"U",mu);
    auto& Umu = compatible<iColourMatrix<vCoeff_t>>(l)->l;
    PokeIndex<LorentzIndex>(U,Umu,mu);
  }

  auto f = new ZMobiusFermion<WI>(U,*grid5,*grid5_rb,*grid4,*grid4_rb,
				  mass,M5,omega,b,c,wp);

  return new cgpt_fermion_operator<ZMobiusFermion<WI>>(f);
}
