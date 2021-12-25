/*
    GPT - Grid Python Toolkit
    Copyright (C) 2021  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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

template<typename T>
void munger_reconstruct_third_row(T* d, T* s, long blocks) {

  const int x=0;
  const int y=1;
  const int z=2;

  thread_for(o, blocks,{
      T* db = &d[o*3*3];
      T* sb = &s[o*3*2];

      for (int i=0;i<3*2;i++)
        db[i] = sb[i];
      
      db[2*3+x] = adj(sb[0*3+y]*sb[1*3+z]-sb[0*3+z]*sb[1*3+y]);
      db[2*3+y] = adj(sb[0*3+z]*sb[1*3+x]-sb[0*3+x]*sb[1*3+z]);
      db[2*3+z] = adj(sb[0*3+x]*sb[1*3+y]-sb[0*3+y]*sb[1*3+x]);
      
    });
}
