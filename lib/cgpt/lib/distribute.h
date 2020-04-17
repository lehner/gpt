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
class cgpt_distribute {
 public:

  static const long SIMD_BASE = 256;
  static long offset(long odx, long idx) { return odx*SIMD_BASE + idx; };
  struct coor { int rank; long offset; };
  struct mp { std::vector<long> src; std::vector<long> dst; };
  struct plan { std::map<int,mp> cr; std::vector<long> tasks; };
  struct data_simd { void* local; long word, Nsimd, simd_word; };

  int rank;

  // word == sizeof(sobj), simd_word == sizeof(Coeff_t)
  cgpt_distribute(int rank, Grid_MPI_Comm comm);

  void create_plan(const std::vector<coor>& c, plan& plan);

  void copy_to(const plan& p,std::vector<data_simd> & src, void* dst);

  void copy_from(const plan& p,void* src, std::vector<data_simd> & dst);

 protected:
  void split(const std::vector<coor>& c, std::map<int,mp>& s);
  Grid_MPI_Comm comm;

  int mpi_ranks, mpi_rank;
  std::vector<int> mpi_rank_map;

  // plan
  void packet_prepare_need(std::vector<long>& data, const std::map<int,mp>& cr);
  void wishlists_to_root(const std::vector<long>& wishlist, std::map<int, std::vector<long> >& wishlists);
  void send_tasks_to_ranks(const std::map<int, std::vector<long> >& wishlists, std::vector<long>& tasks);
  void get_send_tasks_for_rank(int i, const std::map<int, std::vector<long> >& wishlists, std::vector<long>& tasks);

  // copy
  void copy_data(const mp& m, std::vector<data_simd>& src, void* dst);
  void copy_data_rev(const mp& m, void* src, std::vector<data_simd>& dst);
  void copy_remote(const std::vector<long>& tasks, const std::map<int,mp>& cr,std::vector<data_simd>& src, void* dst);
  void copy_remote_rev(const std::vector<long>& tasks, const std::map<int,mp>& cr, void* src,std::vector<data_simd>& dst);

  // util
  long word_total(std::vector<data_simd>& src) {
    long r = 0;
    for (auto& s : src) {
      ASSERT(s.Nsimd <= SIMD_BASE);
      r+=s.word;
    }
    return r;
  }
};
