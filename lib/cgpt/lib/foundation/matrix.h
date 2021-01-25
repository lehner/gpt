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

template<typename vector_object, typename functor_t>
inline void matrixEigenFunctor(PVector<Lattice<vector_object>>&        matrix_result,
			       const PVector<Lattice<vector_object>>&  matrix,
			       long                                    n_virtual,
			       functor_t                               functor) {
  
  ASSERT(matrix_result.size() > 0 && matrix.size() > 0);

  ASSERT(matrix_result.size() % n_virtual == 0);
  ASSERT(matrix_result.size() == n_virtual);
  long matrix_result_n = matrix_result.size() / n_virtual;

  ASSERT(matrix.size() % n_virtual == 0);
  ASSERT(matrix.size() == n_virtual);

  long matrix_n = matrix.size() / n_virtual;

  ASSERT(matrix_result_n == matrix_n);

  conformable(matrix_result[0].Grid(), matrix[0].Grid());
  GridBase *grid = matrix[0].Grid();

  long lsites = grid->lSites();

  typedef typename vector_object::scalar_object scalar_object;

  VECTOR_VIEW_OPEN(matrix_result,matrix_result_v,CpuWrite);
  VECTOR_VIEW_OPEN(matrix,matrix_v,CpuRead);

  eigenConverter<scalar_object> converter(n_virtual);
    
  thread_for(_idx, lsites, { // NOTE: Not on GPU because of Eigen & (peek/poke)LocalSite
    auto site = _idx;

    Eigen::MatrixXcd matrix_eigen = converter.matrix(), matrix_result_eigen = converter.matrix();

    scalar_object matrix_result_tmp = Zero();
    scalar_object matrix_tmp = Zero();

    Coordinate lcoor;
    grid->LocalIndexToLocalCoor(site, lcoor);

    for (long lex_outer=0; lex_outer<n_virtual; lex_outer++) {
      peekLocalSite(matrix_tmp, matrix_v[lex_outer], lcoor);
      converter.fillMatrix(lex_outer, matrix_eigen, matrix_tmp);
    }

    functor(matrix_result_eigen, matrix_eigen);

    for (long lex_outer=0; lex_outer<n_virtual; lex_outer++) {
      converter.fillObject(lex_outer, matrix_result_tmp, matrix_result_eigen);
      pokeLocalSite(matrix_result_tmp, matrix_result_v[lex_outer], lcoor);
    }
  });

  VECTOR_VIEW_CLOSE(matrix_result_v);
  VECTOR_VIEW_CLOSE(matrix_v);
}

template<typename vector_object, typename functor_t>
inline void matrixEigenFunctor(Lattice<iSinglet<typename vector_object::vector_type>>&        matrix_result,
			       const PVector<Lattice<vector_object>>&  matrix,
			       long                                    n_virtual,
			       functor_t                               functor) {
  
  ASSERT(matrix.size() > 0);

  ASSERT(matrix.size() % n_virtual == 0);
  ASSERT(matrix.size() == n_virtual);

  long matrix_n = matrix.size() / n_virtual;

  conformable(matrix_result.Grid(), matrix[0].Grid());
  GridBase *grid = matrix[0].Grid();

  long lsites = grid->lSites();

  typedef typename std::remove_reference<decltype(matrix[0])>::type::scalar_object scalar_object;
  typedef typename std::remove_reference<decltype(matrix_result)>::type::scalar_object       singlet_object;

  autoView(matrix_result_v, matrix_result, CpuWrite);
  VECTOR_VIEW_OPEN(matrix,matrix_v,CpuRead);

  eigenConverter<scalar_object> converter(n_virtual);
    
  thread_for(_idx, lsites, { // NOTE: Not on GPU because of Eigen & (peek/poke)LocalSite
    auto site = _idx;

    Eigen::MatrixXcd matrix_eigen = converter.matrix();

    singlet_object singlet_tmp;
    scalar_object matrix_tmp = Zero();

    Coordinate lcoor;
    grid->LocalIndexToLocalCoor(site, lcoor);

    for (long lex_outer=0; lex_outer<n_virtual; lex_outer++) {
      peekLocalSite(matrix_tmp, matrix_v[lex_outer], lcoor);
      converter.fillMatrix(lex_outer, matrix_eigen, matrix_tmp);
    }

    singlet_tmp()()() = functor(matrix_eigen);
    
    pokeLocalSite(singlet_tmp, matrix_result_v, lcoor);
  });

  VECTOR_VIEW_CLOSE(matrix_v);
}

template<typename vector_object>
inline void invertMatrix(PVector<Lattice<vector_object>>&        matrix_result,
			 const PVector<Lattice<vector_object>>&  matrix,
			 long                                    n_virtual) {

  matrixEigenFunctor(matrix_result, matrix, n_virtual,
		     [](Eigen::MatrixXcd & dst, const Eigen::MatrixXcd & src) {
		       dst = src.inverse();
		     });
}

template<typename vector_object>
inline void determinant(Lattice<iSinglet<typename vector_object::vector_type>>& det,
			const PVector<Lattice<vector_object>>&  matrix,
			long                                    n_virtual) {

  matrixEigenFunctor(det, matrix, n_virtual,
		     [](const Eigen::MatrixXcd & src) -> ComplexD {
		       return src.determinant();
		     });
}
