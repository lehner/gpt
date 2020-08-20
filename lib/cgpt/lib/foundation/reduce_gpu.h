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


    This code is based on original Grid code.
*/

template <class vobj, class sobj,class Iterator>
  __global__ void reduceKernel(const vobj *lat, sobj *buffer, Iterator n, Iterator nvec, Iterator numBlocks, Iterator osites) {
  
  Iterator blockSize = blockDim.x;
  
  // perform reduction for this block and
  // write result to global memory buffer
  for (Integer i=0;i<nvec;i++)
    reduceBlocks(&lat[osites*i], &buffer[i*numBlocks], n);
  
  if (gridDim.x > 1) {
    
    const Iterator tid = threadIdx.x;
    __shared__ bool amLast;
    // force shared memory alignment
    extern __shared__ __align__(COALESCE_GRANULARITY) unsigned char shmem_pointer[];
    // it's not possible to have two extern __shared__ arrays with same name
    // but different types in different scopes -- need to cast each time
    sobj *smem = (sobj *)shmem_pointer;
    
    // wait until all outstanding memory instructions in this thread are finished
    acceleratorFence();
    
    if (tid==0) {
      unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
      // true if this block is the last block to be done
      amLast = (ticket == gridDim.x-1);
    }
    
    // each thread must read the correct value of amLast
    acceleratorSynchroniseAll();

    if (amLast) {
      // reduce buffer[0], ..., buffer[gridDim.x-1]
      
      for (Integer j=0;j<nvec;j++) {
	Iterator i = tid;
	sobj mySum = Zero();
	while (i < gridDim.x) {
	  mySum += buffer[j*numBlocks + i];
	  i += blockSize;
	}
	
	reduceBlock(smem, mySum, tid);
      
	if (tid==0) {
	  buffer[j*numBlocks + 0] = smem[0];
	  // reset count variable
	  retirementCount = 0;
	}
      }
    }
  }
}

template <class vobj, class stype>
inline void sumD_gpu(stype* result, const vobj *lat, Integer osites, Integer nvec) 
{
  typedef typename vobj::scalar_objectD sobj;
  typedef decltype(lat) Iterator;
  
  Integer nsimd= vobj::Nsimd();
  Integer size = osites*nsimd;

  Integer numThreads, numBlocks;
  getNumBlocksAndThreads(size, sizeof(sobj), numThreads, numBlocks);
  Integer smemSize = numThreads * sizeof(sobj);

  Vector<sobj> buffer(numBlocks * nvec);
  sobj *buffer_v = &buffer[0];
  
  reduceKernel<<< numBlocks, numThreads, smemSize >>>(lat, buffer_v, size, nvec, numBlocks, osites);
  accelerator_barrier();

  for (Integer i=0;i<nvec;i++)
    result[i] = buffer_v[i * numBlocks];

}
