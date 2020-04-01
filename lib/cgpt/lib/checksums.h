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
static uint32_t cgpt_crc32(unsigned char* data, int64_t len) {
  // crc32 of zlib was incorrect for very large sizes, so do it block-wise
  uint32_t crc = 0x0;
  off_t blk = 0;
  off_t step = 1024*1024*1024;
  while (len > step) {
    crc = crc32(crc,&data[blk],step);
    blk += step;
    len -= step;
  }
  
  return crc32(crc,&data[blk],len);
}
