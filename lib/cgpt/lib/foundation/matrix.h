/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
                  2020  Daniel Richtmann (daniel.richtmann@ur.de, https://github.com/lehner/gpt)

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

#include <type_traits>
template<class stype, int Ncolour>
typename std::enable_if<isGridFundamental<stype>::value, void>::type
convertToEigen(const iScalar<iScalar<iMatrix<stype, Ncolour>>>& matrix_grid,
               Eigen::MatrixXcd&                                matrix_eigen) {
  for(long row=0;row<Ncolour;row++) {
    for(long col=0;col<Ncolour;col++) {
      matrix_eigen(row, col) = static_cast<ComplexD>(TensorRemove(matrix_grid()()(row, col)));
    }
  }
}
template<class stype, int Nspin>
typename std::enable_if<isGridFundamental<stype>::value, void>::type
convertToEigen(const iScalar<iMatrix<iScalar<stype>, Nspin>>& matrix_grid,
               Eigen::MatrixXcd&                              matrix_eigen) {
  for(long row=0;row<Nspin;row++) {
    for(long col=0;col<Nspin;col++) {
      matrix_eigen(row, col) = static_cast<ComplexD>(TensorRemove(matrix_grid()(row, col)()));
    }
  }
}
template<class stype, int Nspin, int Ncolour>
typename std::enable_if<isGridFundamental<stype>::value, void>::type
convertToEigen(const iScalar<iMatrix<iMatrix<stype, Ncolour>, Nspin>>& matrix_grid,
               Eigen::MatrixXcd&                                       matrix_eigen) {
  for (long s_row=0;s_row<Nspin;s_row++) {
    for (long s_col=0;s_col<Nspin;s_col++) {
      for (long c_row=0;c_row<Ncolour;c_row++) {
        for (long c_col=0;c_col<Ncolour;c_col++) {
          matrix_eigen(s_row*Ncolour+c_row, s_col*Ncolour+c_col) = static_cast<ComplexD>(TensorRemove(matrix_grid()(s_row, s_col)(c_row, c_col)));
        }
      }
    }
  }
}
template<class sobj>
void convertToEigen(const sobj&       matrix_grid,
                    Eigen::MatrixXcd& matrix_eigen) {
  ERR("Not implemented");
}


template<class stype, int Ncolour>
typename std::enable_if<isGridFundamental<stype>::value, void>::type
convertFromEigen(const Eigen::MatrixXcd&                    matrix_eigen,
                 iScalar<iScalar<iMatrix<stype, Ncolour>>>& matrix_grid) {
  for(long row=0;row<Ncolour;row++) {
    for(long col=0;col<Ncolour;col++) {
      matrix_grid()()(row, col) = matrix_eigen(row, col);
    }
  }
}
template<class stype, int Nspin>
typename std::enable_if<isGridFundamental<stype>::value, void>::type
convertFromEigen(const Eigen::MatrixXcd&                  matrix_eigen,
                 iScalar<iMatrix<iScalar<stype>, Nspin>>& matrix_grid) {
  for(long row=0;row<Nspin;row++) {
    for(long col=0;col<Nspin;col++) {
      matrix_grid()(row, col)() = matrix_eigen(row, col);
    }
  }
}
template<class stype, int Nspin, int Ncolour>
typename std::enable_if<isGridFundamental<stype>::value, void>::type
convertFromEigen(const Eigen::MatrixXcd&                           matrix_eigen,
                 iScalar<iMatrix<iMatrix<stype, Ncolour>, Nspin>>& matrix_grid) {
  for (long s_row=0;s_row<Nspin;s_row++) {
    for (long s_col=0;s_col<Nspin;s_col++) {
      for (long c_row=0;c_row<Ncolour;c_row++) {
        for (long c_col=0;c_col<Ncolour;c_col++) {
          matrix_grid()(s_row, s_col)(c_row, c_col) = matrix_eigen(s_row*Ncolour+c_row, s_col*Ncolour+c_col);
        }
      }
    }
  }
}
template<class sobj>
void convertFromEigen(const Eigen::MatrixXcd& matrix_eigen,
                      sobj&                   matrix_grid) {
  ERR("Not implemented");
}


template<class vobj>
inline void invertMatrix(PVector<Lattice<vobj>>&       matrix_inv,
                         const PVector<Lattice<vobj>>& matrix,
                         long                          n_virtual) {

  ASSERT(matrix_inv.size() == 1 && matrix.size() == 1);
  ASSERT(n_virtual == 1);

  conformable(matrix_inv[0], matrix[0]);
  GridBase *grid = matrix[0].Grid();

  long lsites = grid->lSites();

  typedef typename std::remove_reference<decltype(matrix_inv[0])>::type::scalar_object scalar_object;

  VECTOR_VIEW_OPEN(matrix_inv,matrix_inv_v,CpuWrite);
  VECTOR_VIEW_OPEN(matrix,matrix_v,CpuRead);

  const int N = sqrt(GridTypeMapper<vobj>::count); // count returns total number of elems in tensor

  std::cout << GridLogMessage << "Calling generic version of invertMatrix with N = " << N << std::endl;

  thread_for(_idx, lsites, { // NOTE: Not on GPU because of Eigen & (peek/poke)LocalSite
    auto site = _idx;

    Eigen::MatrixXcd matrix_inv_eigen = Eigen::MatrixXcd::Zero(N, N);
    Eigen::MatrixXcd matrix_eigen = Eigen::MatrixXcd::Zero(N, N);

    scalar_object matrix_inv_tmp = Zero();
    scalar_object matrix_tmp = Zero();

    Coordinate lcoor;
    grid->LocalIndexToLocalCoor(site, lcoor);

    peekLocalSite(matrix_tmp, matrix_v[0], lcoor);
    convertToEigen(matrix_tmp, matrix_eigen);

    matrix_inv_eigen = matrix_eigen.inverse();

    convertFromEigen(matrix_inv_eigen, matrix_inv_tmp);
    pokeLocalSite(matrix_inv_tmp, matrix_inv_v[0], lcoor);

  });

  VECTOR_VIEW_CLOSE(matrix_inv_v);
  VECTOR_VIEW_CLOSE(matrix_v);
}

// template<class CComplex,int basis_virtual_size, typename std::enable_if<CComplex::TensorLevel == 3, void>::type* = nullptr>
template<class CComplex,int basis_virtual_size>
inline void invertMatrix(PVector<Lattice<iMatrix<CComplex, basis_virtual_size>>>&        matrix_inv,
			 const PVector<Lattice<iMatrix<CComplex, basis_virtual_size>>>&  matrix,
			 long                                                            n_virtual) {

  ASSERT(matrix_inv.size() > 0 && matrix.size() > 0);

  ASSERT(matrix_inv.size() % n_virtual == 0);
  ASSERT(matrix_inv.size() == n_virtual);
  long matrix_inv_n = matrix_inv.size() / n_virtual;

  ASSERT(matrix.size() % n_virtual == 0);
  ASSERT(matrix.size() == n_virtual);

  long matrix_n = matrix.size() / n_virtual;

  ASSERT(matrix_inv_n == matrix_n);

  conformable(matrix_inv[0].Grid(), matrix[0].Grid());
  GridBase *grid = matrix[0].Grid();

  long lsites = grid->lSites();

  long n_virtual_red = long(sqrt(n_virtual));
  long nbasis_global = n_virtual_red * basis_virtual_size;

  typedef typename iMatrix<CComplex, basis_virtual_size>::scalar_object scalar_object;

  VECTOR_VIEW_OPEN(matrix_inv,matrix_inv_v,CpuWrite);
  VECTOR_VIEW_OPEN(matrix,matrix_v,CpuRead);

  std::cout << GridLogMessage << "Calling dedicated version for virtual fields" << std::endl;

  thread_for(_idx, lsites, { // NOTE: Not on GPU because of Eigen & (peek/poke)LocalSite
    auto site = _idx;

    Eigen::MatrixXcd matrix_inv_eigen = Eigen::MatrixXcd::Zero(nbasis_global, nbasis_global);
    Eigen::MatrixXcd matrix_eigen = Eigen::MatrixXcd::Zero(nbasis_global, nbasis_global);

    scalar_object matrix_inv_tmp = Zero();
    scalar_object matrix_tmp = Zero();

    Coordinate lcoor;
    grid->LocalIndexToLocalCoor(site, lcoor);

    // convention for indices:
    // lex_outer = lexicographic index in array of v_objs
    // row_outer = row index    in array of v_objs (column-major ordering)
    // col_outer = column index in array of v_objs (column-major ordering)
    // row_inner = row index    inside a v_obj = grid matrix tensor (row-major ordering)
    // col_inner = column index inside a v_obj = grid matrix tensor (row-major ordering)
    // row_global = row index    of combination of v_objs viewed as 1 single big matrix tensor (row-major ordering)
    // col_global = column index of combination of v_objs viewed as 1 single big matrix tensor (row-major ordering)

    for (long lex_outer=0; lex_outer<n_virtual; lex_outer++) {
      peekLocalSite(matrix_tmp, matrix_v[lex_outer], lcoor);
      long row_outer = lex_outer % n_virtual_red;
      long col_outer = lex_outer / n_virtual_red;
      for (long row_inner=0; row_inner<basis_virtual_size; row_inner++) {
        for (long col_inner=0; col_inner<basis_virtual_size; col_inner++) {
          long row_global = row_outer * basis_virtual_size + row_inner;
          long col_global = col_outer * basis_virtual_size + col_inner;
          matrix_eigen(row_global, col_global) = static_cast<ComplexD>(TensorRemove(matrix_tmp(row_inner, col_inner)));
        }
      }
    }

    matrix_inv_eigen = matrix_eigen.inverse();

    for (long lex_outer=0; lex_outer<n_virtual; lex_outer++) {
      long row_outer = lex_outer % n_virtual_red;
      long col_outer = lex_outer / n_virtual_red;
      for (long row_inner=0; row_inner<basis_virtual_size; row_inner++) {
        for (long col_inner=0; col_inner<basis_virtual_size; col_inner++) {
          long row_global = row_outer * basis_virtual_size + row_inner;
          long col_global = col_outer * basis_virtual_size + col_inner;
          matrix_inv_tmp(row_inner, col_inner) = matrix_inv_eigen(row_global, col_global);
        }
      }
      pokeLocalSite(matrix_inv_tmp, matrix_inv_v[lex_outer], lcoor);
    }
  });

  VECTOR_VIEW_CLOSE(matrix_inv_v);
  VECTOR_VIEW_CLOSE(matrix_v);
}


template<class vobj>
inline void determinant(Lattice<iSinglet<typename vobj::vector_type>>& det,
                        const PVector<Lattice<vobj>>&                  matrix,
                        long                                           n_virtual) {

  ASSERT(matrix.size() == 1);
  ASSERT(n_virtual == 1);

  conformable(det.Grid(), matrix[0].Grid());
  GridBase *grid = matrix[0].Grid();

  long lsites = grid->lSites();

  typedef typename std::remove_reference<decltype(matrix[0])>::type::scalar_object scalar_object;
  typedef typename std::remove_reference<decltype(det)>::type::scalar_object       singlet_object;

  autoView(det_v, det, CpuWrite);
  VECTOR_VIEW_OPEN(matrix,matrix_v,CpuRead);

  const int N = sqrt(GridTypeMapper<vobj>::count);

  thread_for(_idx, lsites, { // NOTE: Not on GPU because of Eigen & (peek/poke)LocalSite
    auto site = _idx;

    Eigen::MatrixXcd matrix_eigen = Eigen::MatrixXcd::Zero(N, N);

    scalar_object matrix_tmp = Zero();
    singlet_object singlet_tmp;

    Coordinate lcoor;
    grid->LocalIndexToLocalCoor(site, lcoor);

    peekLocalSite(matrix_tmp, matrix_v[0], lcoor);
    convertToEigen(matrix_tmp, matrix_eigen);

    singlet_tmp()()() = matrix_eigen.determinant();
    
    pokeLocalSite(singlet_tmp, det_v, lcoor);

  });

  VECTOR_VIEW_CLOSE(matrix_v);
}
