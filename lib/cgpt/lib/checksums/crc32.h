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
static uint32_t cgpt_crc32(unsigned char* data, int64_t len, uint32_t start_crc = 0) {

  off_t step = 1024*1024*1024;

  if (len == 0)
    return start_crc;

  if (len <= step) {

    //uint32_t ref = crc32(start_crc,data,len);

    // parallel version
    int64_t block_size = 512*1024;
    int64_t blocks = ( len % block_size == 0 ) ? ( len / block_size ) : ( len / block_size + 1 );
    std::vector<uint32_t> pcrcs(blocks);
    thread_for(iblk, blocks, {
	int64_t block_start = block_size * iblk;
	int64_t block_len = std::min(block_size, len - block_start);
	pcrcs[iblk] = crc32(iblk == 0 ? start_crc : 0,&data[block_start],block_len);
    });

    // crc
    uint32_t crc = pcrcs[0];
    // reduce
    for (int iblk=1;iblk<blocks;iblk++) {
      int64_t block_start = block_size * iblk;
      int64_t block_len = std::min(block_size, len - block_start);
      crc = crc32_combine(crc,pcrcs[iblk],block_len);
    }

    //assert(crc == ref);

    return crc;

  } else {

    // crc32 of zlib was incorrect for very large sizes, so do it block-wise
    uint32_t crc = start_crc;
    off_t blk = 0;
    
    while (len > step) {
      crc = cgpt_crc32(&data[blk],step,crc);
      blk += step;
      len -= step;
    }
    
    return cgpt_crc32(&data[blk],len,crc);
  }
}
