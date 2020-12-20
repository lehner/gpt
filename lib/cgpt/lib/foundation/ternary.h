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
    51 Franklin Street, Fifth Floor, Boston, MA 021basis_virtual_size-1n_virtual_red01 USA.
*/

template<typename S, typename T>
inline void cgpt_where(Lattice<T>& answer, const Lattice<S>& question, const Lattice<T>& yes, const Lattice<T>& no) {

  GridBase* grid = answer.Grid();
  conformable(grid, question.Grid());
  conformable(grid, yes.Grid());
  conformable(grid, no.Grid());

  typedef typename Lattice<S>::scalar_object S_sobj;
  typedef typename Lattice<S>::vector_object S_vobj;
  typedef typename Lattice<T>::scalar_object T_sobj;
  typedef typename Lattice<T>::vector_object T_vobj;
  
  autoView(answer_v, answer, CpuWriteDiscard);
  autoView(question_v, question, CpuRead);
  autoView(yes_v, yes, CpuRead);
  autoView(no_v, no, CpuRead);

  auto oSites = grid->oSites();
  auto Nsimd = grid->Nsimd();

  thread_for(i, oSites, {

      ExtractBuffer<S_sobj> vquestion(Nsimd);
      ExtractBuffer<T_sobj> vyes(Nsimd);
      ExtractBuffer<T_sobj> vno(Nsimd);
      ExtractBuffer<T_sobj> vanswer(Nsimd);

      extract<S_vobj,S_sobj>(question_v[i],vquestion);
      extract<T_vobj,T_sobj>(yes_v[i],vyes);
      extract<T_vobj,T_sobj>(no_v[i],vno);

      for(int s=0;s<Nsimd;s++){
	vanswer[s] = (norm2(vquestion[s]) != 0.0) ? vyes[s] : vno[s];
      }

      merge<T_vobj,T_sobj>(answer_v[i],vanswer);
    });

}
