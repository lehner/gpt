/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Daniel Richtmann (daniel.richtmann@ur.de, https://github.com/lehner/gpt)
                  2020  Nils Meyer       (nils.meyer@ur.de,       https://github.com/lehner/gpt)
                  2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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

// see Grid/qcd/action/fermion/WilsonCloverFermion for description
//
// Modifications done here:
//
// Grid: clover term = 12x12 matrix per site
//
// But: Only two diagonal 6x6 hermitian blocks are non-zero (also true in Grid, verified by running)
// Sufficient to store/transfer only the real parts of the diagonal and one triangular part
// 2 * (6 + 15 * 2) = 72 real or 36 complex words to be stored/transfered
//
// Here: Above but diagonal as complex numbers, i.e., need to store/transfer
// 2 * (6 * 2 + 15 * 2) = 84 real or 42 complex words
//
// Words per site and improvement compared to Grid (combined with the input and output spinors:
//
// - Grid:    2*12 + 12*12 = 168 words -> 1.00 x less
// - Minimal: 2*12 + 36    =  60 words -> 2.80 x less
// - Here:    2*12 + 42    =  66 words -> 2.55 x less
//
// These improvements directly translate to wall-clock time
//
// Data layout:
//
// - diagonal and triangle part as separate lattice fields,
//   this was faster than as 1 combined field on all tested machines
// - diagonal: as expected
// - triangle: store upper right triangle in row major order
// - graphical:
//        0  1  2  3  4
//           5  6  7  8
//              9 10 11 = upper right triangle indices
//                12 13
//                   14
//     0
//        1
//           2
//              3       = diagonal indices
//                 4
//                    5
//     0
//     1  5
//     2  6  9          = lower left triangle indices
//     3  7 10 12
//     4  8 11 13 14

template<class Impl>
class FasterWilsonCloverFermion : public WilsonCloverFermion<Impl> {
  /////////////////////////////////////////////
  // Sizes
  /////////////////////////////////////////////

public:

  static constexpr int Nred      = Nc * Nhs;        // 6
  static constexpr int Nblock    = Nhs;             // 2
  static constexpr int Ndiagonal = Nred;            // 6
  static constexpr int Ntriangle = (Nred - 1) * Nc; // 15

  /////////////////////////////////////////////
  // Type definitions
  /////////////////////////////////////////////

public:

  static_assert(Nd == 4 && Nc == 3 && Ns == 4 && Impl::Dimension == 3, "Wrong dimensions");
  INHERIT_IMPL_TYPES(Impl);

  template<typename vtype> using iImplCloverDiagonal = iScalar<iVector<iVector<vtype, Ndiagonal>, Nblock>>;
  template<typename vtype> using iImplCloverTriangle = iScalar<iVector<iVector<vtype, Ntriangle>, Nblock>>;

  typedef iImplCloverDiagonal<Simd> SiteCloverDiagonal;
  typedef iImplCloverTriangle<Simd> SiteCloverTriangle;

  typedef Lattice<SiteCloverDiagonal> CloverDiagonalField;
  typedef Lattice<SiteCloverTriangle> CloverTriangleField;

  typedef WilsonCloverFermion<Impl> WilsonCloverBase;
  typedef typename WilsonCloverBase::SiteCloverType SiteClover;
  typedef typename WilsonCloverBase::CloverFieldType CloverField;

  /////////////////////////////////////////////
  // Member Functions
  /////////////////////////////////////////////

public:

  FasterWilsonCloverFermion(GaugeField& _Umu,
                            GridCartesian& Fgrid,
                            GridRedBlackCartesian& Hgrid,
                            const RealD _mass,
                            const RealD _csw_r = 0.0,
                            const RealD _csw_t = 0.0,
                            const WilsonAnisotropyCoefficients& clover_anisotropy = WilsonAnisotropyCoefficients(),
                            const ImplParams& impl_p = ImplParams())
    : WilsonCloverFermion<Impl>(_Umu, Fgrid, Hgrid, _mass, _csw_r, _csw_t, clover_anisotropy, impl_p)
    , Diagonal(&Fgrid),        Triangle(&Fgrid)
    , DiagonalEven(&Hgrid),    TriangleEven(&Hgrid)
    , DiagonalOdd(&Hgrid),     TriangleOdd(&Hgrid)
    , DiagonalInv(&Fgrid),     TriangleInv(&Fgrid)
    , DiagonalInvEven(&Hgrid), TriangleInvEven(&Hgrid)
    , DiagonalInvOdd(&Hgrid),  TriangleInvOdd(&Hgrid)
  {
    double t0 = usecond();
    convertLayout(this->CloverTerm, Diagonal, Triangle);
    convertLayout(this->CloverTermEven, DiagonalEven, TriangleEven);
    convertLayout(this->CloverTermOdd, DiagonalOdd, TriangleOdd);

    convertLayout(this->CloverTermInv, DiagonalInv, TriangleInv);
    convertLayout(this->CloverTermInvEven, DiagonalInvEven, TriangleInvEven);
    convertLayout(this->CloverTermInvOdd, DiagonalInvOdd, TriangleInvOdd);

    // TODO: set original clover fields to zero size
    // BUT:  keep them around if we want to apply MDeriv
    double t1 = usecond();
    std::cout << GridLogDebug << "FasterWilsonCloverFermion: layout conversions took " << (t1-t0)/1e6 << " seconds" << std::endl;
  }


  void convertLayout(const CloverField& full, CloverDiagonalField& diagonal, CloverTriangleField& triangle) {
    conformable(full.Grid(), diagonal.Grid());
    conformable(full.Grid(), triangle.Grid());

    diagonal.Checkerboard()     = full.Checkerboard();
    triangle.Checkerboard() = full.Checkerboard();

    autoView(full_v,     full,     AcceleratorRead);
    autoView(diagonal_v, diagonal, AcceleratorWrite);
    autoView(triangle_v, triangle, AcceleratorWrite);

    // NOTE: this function cannot be 'private' since nvcc forbids this for kernels
    accelerator_for(ss, full.Grid()->oSites(), 1, {
      for(int s_row = 0; s_row < Ns; s_row++) {
        for(int s_col = 0; s_col < Ns; s_col++) {
          if(abs(s_row - s_col) > 1 || s_row + s_col == 3) continue;
          int block       = s_row / Nhs;
          int s_row_block = s_row % Nhs;
          int s_col_block = s_col % Nhs;
          for(int c_row = 0; c_row < Nc; c_row++) {
            for(int c_col = 0; c_col < Nc; c_col++) {
              int i = s_row_block * Nc + c_row;
              int j = s_col_block * Nc + c_col;
              if(i == j)
                diagonal_v[ss]()(block)(i) = full_v[ss]()(s_row, s_col)(c_row, c_col);
              else if(i < j)
                triangle_v[ss]()(block)(triangle_index(i, j)) = full_v[ss]()(s_row, s_col)(c_row, c_col);
              else
                continue;
            }
          }
        }
      }
    });
  }


  void Mooee(const FermionField& in, FermionField& out) override {
    if(in.Grid()->_isCheckerBoarded) {
      if(in.Checkerboard() == Odd) {
        MooeeInternalImpl(in, out, DiagonalOdd, TriangleOdd);
      } else {
        MooeeInternalImpl(in, out, DiagonalEven, TriangleEven);
      }
    } else {
      MooeeInternalImpl(in, out, Diagonal, Triangle);
    }
  }


  void MooeeDag(const FermionField& in, FermionField& out) override {
    Mooee(in, out); // blocks are hermitian
  }


  void MooeeInv(const FermionField& in, FermionField& out) override {
    if(in.Grid()->_isCheckerBoarded) {
      if(in.Checkerboard() == Odd) {
        MooeeInternalImpl(in, out, DiagonalInvOdd, TriangleInvOdd);
      } else {
        MooeeInternalImpl(in, out, DiagonalInvEven, TriangleInvEven);
      }
    } else {
      MooeeInternalImpl(in, out, DiagonalInv, TriangleInv);
    }
  }


  void MooeeInvDag(const FermionField& in, FermionField& out) override {
    MooeeInv(in, out); // blocks are hermitian
  }


  void MooeeInternalImpl(const FermionField&        in,
                         FermionField&              out,
                         const CloverDiagonalField& diagonal,
                         const CloverTriangleField& triangle) {
    assert(in.Checkerboard() == Odd || in.Checkerboard() == Even);
    out.Checkerboard() = in.Checkerboard();
    conformable(in.Grid(), out.Grid());
    conformable(in.Grid(), diagonal.Grid());
    conformable(in.Grid(), triangle.Grid());

#if defined(GRID_CUDA) || defined(GRID_HIP)
    MooeeKernel_gpu(in.oSites(), in, out, diagonal, triangle);
#else
    MooeeKernel_cpu(in.oSites(), in, out, diagonal, triangle);
#endif
  }


  void MooeeKernel_gpu(int                        Nsite,
                       const FermionField&        in,
                       FermionField&              out,
                       const CloverDiagonalField& diagonal,
                       const CloverTriangleField& triangle) {
    autoView(diagonal_v, diagonal, AcceleratorRead);
    autoView(triangle_v, triangle, AcceleratorRead);
    autoView(in_v,       in,       AcceleratorRead);
    autoView(out_v,      out,      AcceleratorWrite);

    typedef decltype(coalescedRead(out_v[0])) CalcSpinor;

    accelerator_for(ss, Nsite, Simd::Nsimd(), {
      CalcSpinor res;
      CalcSpinor in_t = in_v(ss);
      auto diagonal_t = diagonal_v(ss);
      auto triangle_t = triangle_v(ss);
      for(int block=0; block<Nhs; block++) {
        int s_start = block*Nhs;
        for(int i=0; i<Nred; i++) {
          int si = s_start + i/Nc, ci = i%Nc;
          res()(si)(ci) = diagonal_t()(block)(i) * in_t()(si)(ci);
          for(int j=0; j<Nred; j++) {
            if (j == i) continue;
            int sj = s_start + j/Nc, cj = j%Nc;
            res()(si)(ci) = res()(si)(ci) + triangle_elem(triangle_t, block, i, j) * in_t()(sj)(cj);
          };
        };
      };
      coalescedWrite(out_v[ss], res);
    });
  }


  void MooeeKernel_cpu(int                        Nsite,
                       const FermionField&        in,
                       FermionField&              out,
                       const CloverDiagonalField& diagonal,
                       const CloverTriangleField& triangle) {
    autoView(diagonal_v, diagonal, CpuRead);
    autoView(triangle_v, triangle, CpuRead);
    autoView(in_v,       in,       CpuRead);
    autoView(out_v,      out,      CpuWrite);

    typedef SiteSpinor CalcSpinor;

#if defined(A64FX) || defined(A64FXFIXEDSIZE)
#define PREFETCH_CLOVER(BASE) {                                     \
    uint64_t base;                                                  \
    int pf_dist_L1 = 1;                                             \
    int pf_dist_L2 = -5; /* -> penalty -> disable */                \
                                                                    \
    if ((pf_dist_L1 >= 0) && (ss + pf_dist_L1 < Nsite)) {           \
      base = (uint64_t)&diag_t()(pf_dist_L1+BASE)(0);               \
      svprfd(svptrue_b64(), (int64_t*)(base +    0), SV_PLDL1STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base +  256), SV_PLDL1STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base +  512), SV_PLDL1STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base +  768), SV_PLDL1STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base + 1024), SV_PLDL1STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base + 1280), SV_PLDL1STRM); \
    }                                                               \
                                                                    \
    if ((pf_dist_L2 >= 0) && (ss + pf_dist_L2 < Nsite)) {           \
      base = (uint64_t)&diag_t()(pf_dist_L2+BASE)(0);               \
      svprfd(svptrue_b64(), (int64_t*)(base +    0), SV_PLDL2STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base +  256), SV_PLDL2STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base +  512), SV_PLDL2STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base +  768), SV_PLDL2STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base + 1024), SV_PLDL2STRM); \
      svprfd(svptrue_b64(), (int64_t*)(base + 1280), SV_PLDL2STRM); \
    }                                                               \
  }
// TODO: Implement/generalize this for other architectures
// I played around a bit on KNL (see below) but didn't bring anything
// #elif defined(AVX512)
// #define PREFETCH_CLOVER(BASE) {                              \
//     uint64_t base;                                           \
//     int pf_dist_L1 = 1;                                      \
//     int pf_dist_L2 = +4;                                     \
//                                                              \
//     if ((pf_dist_L1 >= 0) && (ss + pf_dist_L1 < Nsite)) {    \
//       base = (uint64_t)&diag_t()(pf_dist_L1+BASE)(0);        \
//       _mm_prefetch((const char*)(base +    0), _MM_HINT_T0); \
//       _mm_prefetch((const char*)(base +   64), _MM_HINT_T0); \
//       _mm_prefetch((const char*)(base +  128), _MM_HINT_T0); \
//       _mm_prefetch((const char*)(base +  192), _MM_HINT_T0); \
//       _mm_prefetch((const char*)(base +  256), _MM_HINT_T0); \
//       _mm_prefetch((const char*)(base +  320), _MM_HINT_T0); \
//     }                                                        \
//                                                              \
//     if ((pf_dist_L2 >= 0) && (ss + pf_dist_L2 < Nsite)) {    \
//       base = (uint64_t)&diag_t()(pf_dist_L2+BASE)(0);        \
//       _mm_prefetch((const char*)(base +    0), _MM_HINT_T1); \
//       _mm_prefetch((const char*)(base +   64), _MM_HINT_T1); \
//       _mm_prefetch((const char*)(base +  128), _MM_HINT_T1); \
//       _mm_prefetch((const char*)(base +  192), _MM_HINT_T1); \
//       _mm_prefetch((const char*)(base +  256), _MM_HINT_T1); \
//       _mm_prefetch((const char*)(base +  320), _MM_HINT_T1); \
//     }                                                        \
//   }
#else
#define PREFETCH_CLOVER(BASE)
#endif

    thread_for(ss, Nsite, {
      CalcSpinor res;
      CalcSpinor in_t = in_v[ss];
      auto diag_t     = diagonal_v[ss]; // "diag" instead of "diagonal" here to make code below easier to read
      auto triangle_t = triangle_v[ss];

      // upper half
      PREFETCH_CLOVER(0);

      auto in_cc_0_0 = conjugate(in_t()(0)(0)); // Nils: reduces number
      auto in_cc_0_1 = conjugate(in_t()(0)(1)); // of conjugates from
      auto in_cc_0_2 = conjugate(in_t()(0)(2)); // 30 to 20
      auto in_cc_1_0 = conjugate(in_t()(1)(0));
      auto in_cc_1_1 = conjugate(in_t()(1)(1));

      res()(0)(0) =               diag_t()(0)( 0) * in_t()(0)(0)
                  +           triangle_t()(0)( 0) * in_t()(0)(1)
                  +           triangle_t()(0)( 1) * in_t()(0)(2)
                  +           triangle_t()(0)( 2) * in_t()(1)(0)
                  +           triangle_t()(0)( 3) * in_t()(1)(1)
                  +           triangle_t()(0)( 4) * in_t()(1)(2);

      res()(0)(1) =           triangle_t()(0)( 0) * in_cc_0_0;
      res()(0)(1) =               diag_t()(0)( 1) * in_t()(0)(1)
                  +           triangle_t()(0)( 5) * in_t()(0)(2)
                  +           triangle_t()(0)( 6) * in_t()(1)(0)
                  +           triangle_t()(0)( 7) * in_t()(1)(1)
                  +           triangle_t()(0)( 8) * in_t()(1)(2)
                  + conjugate(       res()(0)( 1));

      res()(0)(2) =           triangle_t()(0)( 1) * in_cc_0_0
                  +           triangle_t()(0)( 5) * in_cc_0_1;
      res()(0)(2) =               diag_t()(0)( 2) * in_t()(0)(2)
                  +           triangle_t()(0)( 9) * in_t()(1)(0)
                  +           triangle_t()(0)(10) * in_t()(1)(1)
                  +           triangle_t()(0)(11) * in_t()(1)(2)
                  + conjugate(       res()(0)( 2));

      res()(1)(0) =           triangle_t()(0)( 2) * in_cc_0_0
                  +           triangle_t()(0)( 6) * in_cc_0_1
                  +           triangle_t()(0)( 9) * in_cc_0_2;
      res()(1)(0) =               diag_t()(0)( 3) * in_t()(1)(0)
                  +           triangle_t()(0)(12) * in_t()(1)(1)
                  +           triangle_t()(0)(13) * in_t()(1)(2)
                  + conjugate(       res()(1)( 0));

      res()(1)(1) =           triangle_t()(0)( 3) * in_cc_0_0
                  +           triangle_t()(0)( 7) * in_cc_0_1
                  +           triangle_t()(0)(10) * in_cc_0_2
                  +           triangle_t()(0)(12) * in_cc_1_0;
      res()(1)(1) =               diag_t()(0)( 4) * in_t()(1)(1)
                  +           triangle_t()(0)(14) * in_t()(1)(2)
                  + conjugate(       res()(1)( 1));

      res()(1)(2) =           triangle_t()(0)( 4) * in_cc_0_0
                  +           triangle_t()(0)( 8) * in_cc_0_1
                  +           triangle_t()(0)(11) * in_cc_0_2
                  +           triangle_t()(0)(13) * in_cc_1_0
                  +           triangle_t()(0)(14) * in_cc_1_1;
      res()(1)(2) =               diag_t()(0)( 5) * in_t()(1)(2)
                  + conjugate(       res()(1)( 2));

      vstream(out_v[ss]()(0)(0), res()(0)(0));
      vstream(out_v[ss]()(0)(1), res()(0)(1));
      vstream(out_v[ss]()(0)(2), res()(0)(2));
      vstream(out_v[ss]()(1)(0), res()(1)(0));
      vstream(out_v[ss]()(1)(1), res()(1)(1));
      vstream(out_v[ss]()(1)(2), res()(1)(2));

      // lower half
      PREFETCH_CLOVER(1);

      auto in_cc_2_0 = conjugate(in_t()(2)(0));
      auto in_cc_2_1 = conjugate(in_t()(2)(1));
      auto in_cc_2_2 = conjugate(in_t()(2)(2));
      auto in_cc_3_0 = conjugate(in_t()(3)(0));
      auto in_cc_3_1 = conjugate(in_t()(3)(1));

      res()(2)(0) =               diag_t()(1)( 0) * in_t()(2)(0)
                  +           triangle_t()(1)( 0) * in_t()(2)(1)
                  +           triangle_t()(1)( 1) * in_t()(2)(2)
                  +           triangle_t()(1)( 2) * in_t()(3)(0)
                  +           triangle_t()(1)( 3) * in_t()(3)(1)
                  +           triangle_t()(1)( 4) * in_t()(3)(2);

      res()(2)(1) =           triangle_t()(1)( 0) * in_cc_2_0;
      res()(2)(1) =               diag_t()(1)( 1) * in_t()(2)(1)
                  +           triangle_t()(1)( 5) * in_t()(2)(2)
                  +           triangle_t()(1)( 6) * in_t()(3)(0)
                  +           triangle_t()(1)( 7) * in_t()(3)(1)
                  +           triangle_t()(1)( 8) * in_t()(3)(2)
                  + conjugate(       res()(2)( 1));

      res()(2)(2) =           triangle_t()(1)( 1) * in_cc_2_0
                  +           triangle_t()(1)( 5) * in_cc_2_1;
      res()(2)(2) =               diag_t()(1)( 2) * in_t()(2)(2)
                  +           triangle_t()(1)( 9) * in_t()(3)(0)
                  +           triangle_t()(1)(10) * in_t()(3)(1)
                  +           triangle_t()(1)(11) * in_t()(3)(2)
                  + conjugate(       res()(2)( 2));

      res()(3)(0) =           triangle_t()(1)( 2) * in_cc_2_0
                  +           triangle_t()(1)( 6) * in_cc_2_1
                  +           triangle_t()(1)( 9) * in_cc_2_2;
      res()(3)(0) =               diag_t()(1)( 3) * in_t()(3)(0)
                  +           triangle_t()(1)(12) * in_t()(3)(1)
                  +           triangle_t()(1)(13) * in_t()(3)(2)
                  + conjugate(       res()(3)( 0));

      res()(3)(1) =           triangle_t()(1)( 3) * in_cc_2_0
                  +           triangle_t()(1)( 7) * in_cc_2_1
                  +           triangle_t()(1)(10) * in_cc_2_2
                  +           triangle_t()(1)(12) * in_cc_3_0;
      res()(3)(1) =               diag_t()(1)( 4) * in_t()(3)(1)
                  +           triangle_t()(1)(14) * in_t()(3)(2)
                  + conjugate(       res()(3)( 1));

      res()(3)(2) =           triangle_t()(1)( 4) * in_cc_2_0
                  +           triangle_t()(1)( 8) * in_cc_2_1
                  +           triangle_t()(1)(11) * in_cc_2_2
                  +           triangle_t()(1)(13) * in_cc_3_0
                  +           triangle_t()(1)(14) * in_cc_3_1;
      res()(3)(2) =               diag_t()(1)( 5) * in_t()(3)(2)
                  + conjugate(       res()(3)( 2));

      vstream(out_v[ss]()(2)(0), res()(2)(0));
      vstream(out_v[ss]()(2)(1), res()(2)(1));
      vstream(out_v[ss]()(2)(2), res()(2)(2));
      vstream(out_v[ss]()(3)(0), res()(3)(0));
      vstream(out_v[ss]()(3)(1), res()(3)(1));
      vstream(out_v[ss]()(3)(2), res()(3)(2));
    });
  }

  /////////////////////////////////////////////
  // Helpers
  /////////////////////////////////////////////

private:

  template<typename vobj>
  accelerator_inline vobj triangle_elem(const iImplCloverTriangle<vobj>& triangle, int block, int i, int j) {
    assert(i != j);
    if(i < j) {
      return triangle()(block)(triangle_index(i, j));
    } else { // i > j
      return conjugate(triangle()(block)(triangle_index(i, j)));
    }
  }


  accelerator_inline int triangle_index(int i, int j) {
    if(i == j)
      return 0;
    else if(i < j)
      return Nred * (Nred - 1) / 2 - (Nred - i) * (Nred - i - 1) / 2 + j - i - 1;
    else // i > j
      return Nred * (Nred - 1) / 2 - (Nred - j) * (Nred - j - 1) / 2 + i - j - 1;
  }

  /////////////////////////////////////////////
  // Member Data
  /////////////////////////////////////////////

private:

  CloverDiagonalField Diagonal,    DiagonalEven,    DiagonalOdd;
  CloverDiagonalField DiagonalInv, DiagonalInvEven, DiagonalInvOdd;

  CloverTriangleField Triangle,    TriangleEven,    TriangleOdd;
  CloverTriangleField TriangleInv, TriangleInvEven, TriangleInvOdd;
};
