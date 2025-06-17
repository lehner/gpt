/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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
#include "lib.h"
#include "benchmarks.h"

/*
template<typename vtype>
void cgpt_tensor_contract(vtype* C, vtype* A, vtype* B,
			  uint64_t n, uint64_t m, uint64_t k,
			  uint64_t a_ostride, uint64_t a_offset, uint64_t a_istride,
			  uint64_t b_ostride, uint64_t b_offset, uint64_t b_istride,
			  uint64_t c_ostride, uint64_t c_offset, uint64_t c_istride) {
  // take MM as example
  // C[i + j*ldc] = sum(A[i + l*lda] * B[l + j*ldb], l)
  accelerator_for(i, n, {
      uint64_t a0 = i*a_ostride + a_offset;
      uint64_t b0 = i*b_ostride + b_offset;
      uint64_t c0 = i*c_ostride + c_offset;
      typedef decltype(acceleratorRead(A[0])) t;
      for (uint64_t l=0;l<k;l++) {
	t = Zero();
	for (uint64_t j=0;j<m;j++)
	  t += acceleratorRead(A[a0 + a_istride*j]) * acceleratorRead(B[b0 + b_istride*j]);
	acceleratorWrite(C[c0 + l*c_istride], t);
      }
    });
}
*/			  
EXPORT(benchmarks,{
    
    //mask();
    //half();
    //benchmarks(8);
    //benchmarks(16);
    //benchmarks(32);

    GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(4,vComplexD::Nsimd()), GridDefaultMpi());
    
    LatticeSpinColourMatrixD data1(UGrid);

    //typedef typename LatticeSpinColourMatrixD::vector_object vobj;
    //std::vector<
    //cgpt_get_shape<vobj>()
		       
    LatticeSpinColourMatrixD data2(UGrid);
    LatticeComplexD index(UGrid);
    index = Zero();
    std::vector<typename LatticeSpinColourMatrixD::scalar_object> res(192);
    PVector<LatticeSpinColourMatrixD> v;
    v.push_back(&data1);
    v.push_back(&data2);
    cgpt_rank_indexed_sum(v, index, res);

    return PyLong_FromLong(0);
  });

//#include <Grid/qcd/smearing/GaugeConfigurationMasked.h>
//#include <Grid/qcd/smearing/JacobianAction.h>
//using namespace Grid;

EXPORT(test_grid,{
    PyObject* _fields;
    if (!PyArg_ParseTuple(args, "O", &_fields)) {
      return NULL;
    }

#if 0
    std::vector<cgpt_Lattice_base*> fields;
    cgpt_basis_fill(fields,_fields);

    PVector<LatticeColourMatrixD> cfields;
    cgpt_basis_fill(cfields,fields);

    GridCartesian* grid = (GridCartesian*)cfields[0].Grid();
    LatticeGaugeFieldD Umu(grid);

    LatticeGaugeFieldD UmuSm(grid);

    LatticeGaugeFieldD Force(grid);

    for(int mu=0;mu<4;mu++) {
      PokeIndex<LorentzIndex>(Umu,cfields[mu],mu);
      PokeIndex<LorentzIndex>(UmuSm,cfields[mu + 4],mu);
      PokeIndex<LorentzIndex>(Force,cfields[mu + 8],mu);
    }

    typedef PeriodicGaugeImpl<GimplTypesD> PeriodicGimplD; 
    std::cout << GridLogMessage << "P = " << WilsonLoops<PeriodicGimplD>::avgPlaquette(Umu) << std::endl;
    std::cout << GridLogMessage << "Psm = " << WilsonLoops<PeriodicGimplD>::avgPlaquette(UmuSm) << std::endl;

    // next perform smearing here and compare smeared fields
    double rho = 0.124;
    int Nstep  = 8;
    typedef GenericHMCRunner<MinimumNorm2> HMCWrapper;
    Smear_Stout<HMCWrapper::ImplPolicy> Stout(rho);
    SmearedConfigurationMasked<HMCWrapper::ImplPolicy> SmearingPolicy(grid, Nstep, Stout);

    //SmearingPolicy
    SmearingPolicy.fill_smearedSet(Umu);
    LatticeGaugeFieldD gridUmuSm = SmearingPolicy.get_smeared_conf(7);
    std::cout << GridLogMessage << "PsmGrid = " << WilsonLoops<PeriodicGimplD>::avgPlaquette(gridUmuSm) << std::endl;

    std::cout << GridLogMessage << "Test smeared gauge config:" << norm2(closure(gridUmuSm - UmuSm)) / norm2(UmuSm) << std::endl;

    //std::cout << GridLogMessage << "Log Det Jac lvl 7:" << std::setprecision(15) << SmearingPolicy.logDetJacobian() << std::endl;

    RealD ln_det = 0;
    for (int ismr = 7; ismr > 0; --ismr) {
      RealD d = SmearingPolicy.logDetJacobianLevel(SmearingPolicy.get_smeared_conf(ismr-1),ismr);
      std::cout << GridLogMessage << "Log Det Jac from level" << ismr << " = " << std::setprecision(15) << d << std::endl;
      ln_det+= d;
    }
    {
      RealD d = SmearingPolicy.logDetJacobianLevel(*(SmearingPolicy.ThinLinks),0);
      std::cout << GridLogMessage << "Log Det Jac from level" << 0 << " = " << std::setprecision(15) << d << std::endl;
      ln_det += d;
    }

    std::cout << GridLogMessage << "Log Det Jac:" << std::setprecision(15) << -ln_det << std::endl;

    LatticeGaugeFieldD gridForce(grid);
    SmearingPolicy.logDetJacobianForce(gridForce);

    //Dump(gridForce,"grid force");
    //Dump(Force,"gpt force");
    
    std::cout << GridLogMessage << "Test force:" << norm2(closure(gridForce - Force)) / norm2(Force) << std::endl;

    for(int mu=0;mu<4;mu++) {
      cfields[mu + 8] = PeekIndex<LorentzIndex>(gridForce,mu);
    }

#elif 0

    std::vector<cgpt_Lattice_base*> fields;
    cgpt_basis_fill(fields,_fields);

    PVector<LatticeColourMatrixD> cfields;
    cgpt_basis_fill(cfields,fields);

    GridCartesian* grid = (GridCartesian*)cfields[0].Grid();
    LatticeGaugeFieldD Umu(grid);

    for(int mu=0;mu<4;mu++) {
      PokeIndex<LorentzIndex>(Umu,cfields[mu],mu);
    }

    ScidacWriter w(grid->IsBoss());
    w.open("grid.lime");
    emptyUserRecord record;
    w.writeScidacFieldRecord(Umu, record, 0, Grid::BinaryIO::BINARYIO_LEXICOGRAPHIC);
    w.close();

#elif 0
    
    std::vector<cgpt_Lattice_base*> fields;
    cgpt_basis_fill(fields,_fields);

    PVector<LatticeColourMatrixD> cfields;
    cgpt_basis_fill(cfields,fields);

    GridCartesian* grid = (GridCartesian*)cfields[0].Grid();
    LatticeGaugeFieldD Umu(grid);
    LatticeGaugeFieldD UmuR(grid);

    for(int mu=0;mu<4;mu++) {
      PokeIndex<LorentzIndex>(Umu,cfields[mu],mu);
    }

    ScidacReader w;
    w.open("test");
    emptyUserRecord record;
    w.readScidacFieldRecord(UmuR, record);
    w.close();

    std::cout << GridLogMessage << "Test:" << norm2(closure(UmuR - Umu)) << std::endl;

#endif
    return PyLong_FromLong(0);
  });

EXPORT(tests,{
    test_global_memory_system();
    return PyLong_FromLong(0);
  });
