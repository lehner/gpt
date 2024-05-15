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

EXPORT(benchmarks,{
    
    //mask();
    //half();
    //benchmarks(8);
    //benchmarks(16);
    //benchmarks(32);

    GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(4,vComplexD::Nsimd()), GridDefaultMpi());
    
    LatticeSpinColourMatrixD data1(UGrid);
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

EXPORT(tests,{
    test_global_memory_system();
    return PyLong_FromLong(0);
  });
