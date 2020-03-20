/*
  CGPT

  Authors: Christoph Lehner 2020
*/
template<typename vCoeff_t>
cgpt_fermion_operator_base* cgpt_create_wilson_clover(PyObject* args) {

  typedef WilsonImpl<vCoeff_t, FundamentalRepresentation, CoeffReal > WI;
  typedef typename vCoeff_t::scalar_type Coeff_t;

  WilsonAnisotropyCoefficients wac;
  typename WI::ImplParams wp;

  auto grid = get_pointer<GridCartesian>(args,"grid");
  auto grid_rb = get_pointer<GridRedBlackCartesian>(args,"grid_rb");
  RealD mass = get_float(args,"mass");
  RealD csw_r = get_float(args,"csw_r");
  RealD csw_t = get_float(args,"csw_t");
  wac.isAnisotropic = get_bool(args,"isAnisotropic");
  wac.xi_0 = get_float(args,"xi_0");
  wac.nu = get_float(args,"nu");
  wp.boundary_phases = get_complex_vec<Nd>(args,"boundary_phases");

  // Q =   1 + (Nd-1)/xi_0 + m
  //     + W_t + (nu/xi_0) * W_s
  //     - 1/2*[ csw_t * sum_s (sigma_ts F_ts) + (csw_s/xi_0) * sum_ss (sigma_ss F_ss)  ]

  wp.overlapCommsCompute = true;

  Lattice< iLorentzColourMatrix< vCoeff_t > > U(grid);
  for (int mu=0;mu<Nd;mu++) {
    auto l = get_pointer<cgpt_Lattice_base>(args,"U",mu);
    auto& Umu = compatible<iColourMatrix<vCoeff_t>>(l)->l;
    PokeIndex<LorentzIndex>(U,Umu,mu);
  }

  auto f = new WilsonCloverFermion<WI>(U,*grid,*grid_rb,mass,csw_r,csw_t,wac,wp);

  return new cgpt_fermion_operator<WilsonCloverFermion<WI>>(f);
}
