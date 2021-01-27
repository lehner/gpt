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

#include "distribute/view.h"
#include "distribute/bcopy.h"
#include "distribute/global_transfer.h"
#include "distribute/global_memory_transfer.h"

template class global_transfer<int>;
template class global_memory_view<uint64_t,int,uint32_t>;
template class global_memory_transfer<uint64_t,int,uint32_t>;

void test_global_memory_system() {

  int rank = CartesianCommunicator::RankWorld();
  gm_transfer plan(rank, CartesianCommunicator::communicator_world);
  gm_transfer plan_host_buf(rank, CartesianCommunicator::communicator_world);
  int ranks = plan.mpi_ranks;
  //printf("Rank %d/%d here\n",rank,ranks);
  
  gm_view osrc, odst;
  
  size_t word = sizeof(double);
  size_t word_half = word/2;
  size_t nwords = 512;
  size_t nindex = 6;
  
  // every node requests a specific
  srand (time(NULL));
  std::vector<int> src_ranks(nwords*nindex);
  std::vector<int> src_offset(nwords*nindex);
  std::vector<int> src_index(nwords*nindex);
  
  for (int i=0;i<nindex*nwords;i++) {
    src_ranks[i] = rand() % ranks;
    src_offset[i] = rand() % nwords;
    src_index[i] = rand() % nindex;
  }

  //std::cout << GridLogMessage << "Test setup:" << src_ranks << std::endl << src_offset << std::endl << src_index << std::endl;

  osrc.block_size = (rand() % 2 == 0) ? word_half : word;
  odst.block_size = (rand() % 2 == 0) ? word_half : word;

  size_t d = 0, s = 0;
  osrc.blocks.resize(nindex * nwords * (word / osrc.block_size));
  odst.blocks.resize(nindex * nwords * (word / odst.block_size));
  for (int i=0;i<nindex;i++) {
    for (int j=0;j<nwords;j++) {
      int rs = src_ranks[j + i*nwords];
      int js = src_offset[j + i*nwords];
      int is = src_index[j + i*nwords];
      if (osrc.block_size == word_half) {
	osrc.blocks[s++] = { rs, (uint32_t)is, js*word }; // rank, index, offset, size
	osrc.blocks[s++] = { rs, (uint32_t)is, js*word + word_half }; // rank, index, offset, size
      } else {
	osrc.blocks[s++] = { rs, (uint32_t)is, js*word }; // rank, index, offset, size
      }
      if (odst.block_size == word_half) {
	odst.blocks[d++] = { rank, (uint32_t)i, j*word }; // rank, index, offset, size
	odst.blocks[d++] = { rank, (uint32_t)i, j*word + word_half }; // rank, index, offset, size
      } else {
	odst.blocks[d++] = { rank, (uint32_t)i, j*word }; // rank, index, offset, size
      }
    }
  }

  plan.create(odst, osrc, mt_none);
  plan_host_buf.create(odst, osrc, mt_host);
  
  // prepare test data and execute
  std::vector< std::vector<double> > host_src(nindex);
  std::vector< std::vector<double> > host_dst(nindex);
  for (int i=0;i<nindex;i++) {
    host_src[i].resize(nwords);
    host_dst[i].resize(nwords);
    for (int j=0;j<nwords;j++)
      host_src[i][j] = rank * 1000 + j + 100000 * i;
  }
  
  std::vector<gm_transfer::memory_view> dst, src;
  for (int i=0;i<nindex;i++) {
    dst.push_back( { mt_host,&host_dst[i][0],nwords*sizeof(double)} );
    src.push_back( { mt_host,&host_src[i][0],nwords*sizeof(double)} );
  }

  for (int iter=0;iter<2;iter++) {

    for (int i=0;i<nindex;i++) {
      for (int j=0;j<nwords;j++) {
	host_dst[i][j] = -0.1;
      }
    }
    
    if (iter == 0)
      plan.execute(dst,src);
    else
      plan_host_buf.execute(dst,src);

    // test
    for (int i=0;i<nindex;i++) {
      for (int j=0;j<nwords;j++) {
	int rs = src_ranks[j + i*nwords];
	int js = src_offset[j + i*nwords];
	int is = src_index[j + i*nwords];
	
	double expected = rs * 1000 + js + 100000 * is;
	double have = host_dst[i][j];
	if (have != expected) {
	  printf("ITER%d, Rank %d has an error %g != %g (%d %d)\n",iter,rank,expected,have,
		 i,j);
	}
      }
    }
  }

  std::cout << GridLogMessage << "Test of global memory system completed" << std::endl;

}
