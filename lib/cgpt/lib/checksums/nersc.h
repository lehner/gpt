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
static uint32_t nersc_checksum_combine(uint32_t cs, uint32_t csp, int64_t len) {
  return cs + csp;
}

static uint32_t nersc_checksum(uint32_t start_cs, void* pdata, int64_t len) {
  ASSERT(len % 4 == 0);
  len /= 4;
  uint32_t* c = (uint32_t*)pdata;
  for (int64_t i=0;i<len;i++)
    start_cs += c[i];
  return start_cs;
}

static uint32_t cgpt_nersc_checksum(unsigned char* data, int64_t len, uint32_t start_cs = 0) {

  off_t step = 1024*1024*1024;

  if (len == 0)
    return start_cs;

  if (len <= step) {

    // parallel version
    int64_t block_size = 512*1024;
    int64_t blocks = ( len % block_size == 0 ) ? ( len / block_size ) : ( len / block_size + 1 );
    std::vector<uint32_t> pcss(blocks);
    thread_for(iblk, blocks, {
	int64_t block_start = block_size * iblk;
	int64_t block_len = std::min(block_size, len - block_start);
	pcss[iblk] = nersc_checksum(iblk == 0 ? start_cs : 0,&data[block_start],block_len);
    });

    // cs
    uint32_t cs = pcss[0];
    // reduce
    for (int iblk=1;iblk<blocks;iblk++) {
      int64_t block_start = block_size * iblk;
      int64_t block_len = std::min(block_size, len - block_start);
      cs = nersc_checksum_combine(cs,pcss[iblk],block_len);
    }

    return cs;

  } else {

    uint32_t cs = start_cs;
    off_t blk = 0;
    
    while (len > step) {
      cs = cgpt_nersc_checksum(&data[blk],step,cs);
      blk += step;
      len -= step;
    }
    
    return cgpt_nersc_checksum(&data[blk],len,cs);
  }
}
