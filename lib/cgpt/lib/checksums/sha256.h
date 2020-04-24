/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Stephan Brumme
                        Luchang Jin
			Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)


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

    Code in this file is originally from Stephan Brumme, compatible with GPLv2:

        http://create.stephan-brumme.com/disclaimer.html

    and modified by Luchang Jin for https://github.com/waterret/Qlattice/ .

    The original license used by Stephan Brumme is repeated here:

    This software is provided 'as-is', without any express or implied
    warranty.  In no event will the author be held liable for any
    damages arising from the use of this software.  Permission is
    granted to anyone to use this software for any purpose, including
    commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions: The origin of this
    software must not be misrepresented; you must not claim that you
    wrote the original software.  If you use this software in a
    product, an acknowledgment in the product documentation would be
    appreciated but is not required.  Altered source versions must be
    plainly marked as such, and must not be misrepresented as being
    the original software.
*/
static void cgpt_sha256_setInitialHash(uint32_t hash[8]) {
  hash[0] = 0x6a09e667;
  hash[1] = 0xbb67ae85;
  hash[2] = 0x3c6ef372;
  hash[3] = 0xa54ff53a;
  hash[4] = 0x510e527f;
  hash[5] = 0x9b05688c;
  hash[6] = 0x1f83d9ab;
  hash[7] = 0x5be0cd19;
}

static const size_t cgpt_sha256_BlockSize = 512 / 8;

static const size_t cgpt_sha256_HashBytes = 32;

static const size_t cgpt_sha256_HashValues = cgpt_sha256_HashBytes / 4;

static uint32_t cgpt_sha256_rotate(uint32_t a, uint32_t c) {
  return (a >> c) | (a << (32 - c));
}

static uint32_t cgpt_sha256_swap(uint32_t x) {
  return (x >> 24) | ((x >> 8) & 0x0000FF00) | ((x << 8) & 0x00FF0000) |
    (x << 24);
}

static uint32_t cgpt_sha256_f1(uint32_t e, uint32_t f, uint32_t g)
// mix functions for cgpt_sha256_processBlock()
{
  uint32_t term1 = cgpt_sha256_rotate(e, 6) ^ cgpt_sha256_rotate(e, 11) ^ cgpt_sha256_rotate(e, 25);
  uint32_t term2 = (e & f) ^ (~e & g);  //(g ^ (e & (f ^ g)))
  return term1 + term2;
}

static uint32_t cgpt_sha256_f2(uint32_t a, uint32_t b, uint32_t c)
// mix functions for cgpt_sha256_processBlock()
{
  uint32_t term1 = cgpt_sha256_rotate(a, 2) ^ cgpt_sha256_rotate(a, 13) ^ cgpt_sha256_rotate(a, 22);
  uint32_t term2 = ((a | b) & c) | (a & b);  //(a & (b ^ c)) ^ (b & c);
  return term1 + term2;
}

static void cgpt_sha256_processBlock(uint32_t newHash[8], const uint32_t oldHash[8],
				     const uint8_t data[64])
// process 64 bytes of data
// newHash and oldHash and be the same
{
  // get last hash
  uint32_t a = oldHash[0];
  uint32_t b = oldHash[1];
  uint32_t c = oldHash[2];
  uint32_t d = oldHash[3];
  uint32_t e = oldHash[4];
  uint32_t f = oldHash[5];
  uint32_t g = oldHash[6];
  uint32_t h = oldHash[7];
  // data represented as 16x 32-bit words
  const uint32_t* input = (uint32_t*)data;
  // convert to big endian
  uint32_t words[64];
  int i;
  for (i = 0; i < 16; i++) {
#if defined(__BYTE_ORDER) && (__BYTE_ORDER != 0) && \
  (__BYTE_ORDER == __BIG_ENDIAN)
    words[i] = input[i];
#else
    words[i] = cgpt_sha256_swap(input[i]);
#endif
  }
  uint32_t x, y;  // temporaries
  // first round
  x = h + cgpt_sha256_f1(e, f, g) + 0x428a2f98 + words[0];
  y = cgpt_sha256_f2(a, b, c);
  d += x;
  h = x + y;
  x = g + cgpt_sha256_f1(d, e, f) + 0x71374491 + words[1];
  y = cgpt_sha256_f2(h, a, b);
  c += x;
  g = x + y;
  x = f + cgpt_sha256_f1(c, d, e) + 0xb5c0fbcf + words[2];
  y = cgpt_sha256_f2(g, h, a);
  b += x;
  f = x + y;
  x = e + cgpt_sha256_f1(b, c, d) + 0xe9b5dba5 + words[3];
  y = cgpt_sha256_f2(f, g, h);
  a += x;
  e = x + y;
  x = d + cgpt_sha256_f1(a, b, c) + 0x3956c25b + words[4];
  y = cgpt_sha256_f2(e, f, g);
  h += x;
  d = x + y;
  x = c + cgpt_sha256_f1(h, a, b) + 0x59f111f1 + words[5];
  y = cgpt_sha256_f2(d, e, f);
  g += x;
  c = x + y;
  x = b + cgpt_sha256_f1(g, h, a) + 0x923f82a4 + words[6];
  y = cgpt_sha256_f2(c, d, e);
  f += x;
  b = x + y;
  x = a + cgpt_sha256_f1(f, g, h) + 0xab1c5ed5 + words[7];
  y = cgpt_sha256_f2(b, c, d);
  e += x;
  a = x + y;
  // secound round
  x = h + cgpt_sha256_f1(e, f, g) + 0xd807aa98 + words[8];
  y = cgpt_sha256_f2(a, b, c);
  d += x;
  h = x + y;
  x = g + cgpt_sha256_f1(d, e, f) + 0x12835b01 + words[9];
  y = cgpt_sha256_f2(h, a, b);
  c += x;
  g = x + y;
  x = f + cgpt_sha256_f1(c, d, e) + 0x243185be + words[10];
  y = cgpt_sha256_f2(g, h, a);
  b += x;
  f = x + y;
  x = e + cgpt_sha256_f1(b, c, d) + 0x550c7dc3 + words[11];
  y = cgpt_sha256_f2(f, g, h);
  a += x;
  e = x + y;
  x = d + cgpt_sha256_f1(a, b, c) + 0x72be5d74 + words[12];
  y = cgpt_sha256_f2(e, f, g);
  h += x;
  d = x + y;
  x = c + cgpt_sha256_f1(h, a, b) + 0x80deb1fe + words[13];
  y = cgpt_sha256_f2(d, e, f);
  g += x;
  c = x + y;
  x = b + cgpt_sha256_f1(g, h, a) + 0x9bdc06a7 + words[14];
  y = cgpt_sha256_f2(c, d, e);
  f += x;
  b = x + y;
  x = a + cgpt_sha256_f1(f, g, h) + 0xc19bf174 + words[15];
  y = cgpt_sha256_f2(b, c, d);
  e += x;
  a = x + y;
  // extend to 24 words
  for (; i < 24; i++)
    words[i] = words[i - 16] +
      (cgpt_sha256_rotate(words[i - 15], 7) ^ cgpt_sha256_rotate(words[i - 15], 18) ^
       (words[i - 15] >> 3)) +
               words[i - 7] +
      (cgpt_sha256_rotate(words[i - 2], 17) ^ cgpt_sha256_rotate(words[i - 2], 19) ^
       (words[i - 2] >> 10));
  // third round
  x = h + cgpt_sha256_f1(e, f, g) + 0xe49b69c1 + words[16];
  y = cgpt_sha256_f2(a, b, c);
  d += x;
  h = x + y;
  x = g + cgpt_sha256_f1(d, e, f) + 0xefbe4786 + words[17];
  y = cgpt_sha256_f2(h, a, b);
  c += x;
  g = x + y;
  x = f + cgpt_sha256_f1(c, d, e) + 0x0fc19dc6 + words[18];
  y = cgpt_sha256_f2(g, h, a);
  b += x;
  f = x + y;
  x = e + cgpt_sha256_f1(b, c, d) + 0x240ca1cc + words[19];
  y = cgpt_sha256_f2(f, g, h);
  a += x;
  e = x + y;
  x = d + cgpt_sha256_f1(a, b, c) + 0x2de92c6f + words[20];
  y = cgpt_sha256_f2(e, f, g);
  h += x;
  d = x + y;
  x = c + cgpt_sha256_f1(h, a, b) + 0x4a7484aa + words[21];
  y = cgpt_sha256_f2(d, e, f);
  g += x;
  c = x + y;
  x = b + cgpt_sha256_f1(g, h, a) + 0x5cb0a9dc + words[22];
  y = cgpt_sha256_f2(c, d, e);
  f += x;
  b = x + y;
  x = a + cgpt_sha256_f1(f, g, h) + 0x76f988da + words[23];
  y = cgpt_sha256_f2(b, c, d);
  e += x;
  a = x + y;
  // extend to 32 words
  for (; i < 32; i++)
    words[i] = words[i - 16] +
      (cgpt_sha256_rotate(words[i - 15], 7) ^ cgpt_sha256_rotate(words[i - 15], 18) ^
       (words[i - 15] >> 3)) +
               words[i - 7] +
      (cgpt_sha256_rotate(words[i - 2], 17) ^ cgpt_sha256_rotate(words[i - 2], 19) ^
       (words[i - 2] >> 10));
  // fourth round
  x = h + cgpt_sha256_f1(e, f, g) + 0x983e5152 + words[24];
  y = cgpt_sha256_f2(a, b, c);
  d += x;
  h = x + y;
  x = g + cgpt_sha256_f1(d, e, f) + 0xa831c66d + words[25];
  y = cgpt_sha256_f2(h, a, b);
  c += x;
  g = x + y;
  x = f + cgpt_sha256_f1(c, d, e) + 0xb00327c8 + words[26];
  y = cgpt_sha256_f2(g, h, a);
  b += x;
  f = x + y;
  x = e + cgpt_sha256_f1(b, c, d) + 0xbf597fc7 + words[27];
  y = cgpt_sha256_f2(f, g, h);
  a += x;
  e = x + y;
  x = d + cgpt_sha256_f1(a, b, c) + 0xc6e00bf3 + words[28];
  y = cgpt_sha256_f2(e, f, g);
  h += x;
  d = x + y;
  x = c + cgpt_sha256_f1(h, a, b) + 0xd5a79147 + words[29];
  y = cgpt_sha256_f2(d, e, f);
  g += x;
  c = x + y;
  x = b + cgpt_sha256_f1(g, h, a) + 0x06ca6351 + words[30];
  y = cgpt_sha256_f2(c, d, e);
  f += x;
  b = x + y;
  x = a + cgpt_sha256_f1(f, g, h) + 0x14292967 + words[31];
  y = cgpt_sha256_f2(b, c, d);
  e += x;
  a = x + y;
  // extend to 40 words
  for (; i < 40; i++)
    words[i] = words[i - 16] +
      (cgpt_sha256_rotate(words[i - 15], 7) ^ cgpt_sha256_rotate(words[i - 15], 18) ^
       (words[i - 15] >> 3)) +
               words[i - 7] +
      (cgpt_sha256_rotate(words[i - 2], 17) ^ cgpt_sha256_rotate(words[i - 2], 19) ^
       (words[i - 2] >> 10));
  // fifth round
  x = h + cgpt_sha256_f1(e, f, g) + 0x27b70a85 + words[32];
  y = cgpt_sha256_f2(a, b, c);
  d += x;
  h = x + y;
  x = g + cgpt_sha256_f1(d, e, f) + 0x2e1b2138 + words[33];
  y = cgpt_sha256_f2(h, a, b);
  c += x;
  g = x + y;
  x = f + cgpt_sha256_f1(c, d, e) + 0x4d2c6dfc + words[34];
  y = cgpt_sha256_f2(g, h, a);
  b += x;
  f = x + y;
  x = e + cgpt_sha256_f1(b, c, d) + 0x53380d13 + words[35];
  y = cgpt_sha256_f2(f, g, h);
  a += x;
  e = x + y;
  x = d + cgpt_sha256_f1(a, b, c) + 0x650a7354 + words[36];
  y = cgpt_sha256_f2(e, f, g);
  h += x;
  d = x + y;
  x = c + cgpt_sha256_f1(h, a, b) + 0x766a0abb + words[37];
  y = cgpt_sha256_f2(d, e, f);
  g += x;
  c = x + y;
  x = b + cgpt_sha256_f1(g, h, a) + 0x81c2c92e + words[38];
  y = cgpt_sha256_f2(c, d, e);
  f += x;
  b = x + y;
  x = a + cgpt_sha256_f1(f, g, h) + 0x92722c85 + words[39];
  y = cgpt_sha256_f2(b, c, d);
  e += x;
  a = x + y;
  // extend to 48 words
  for (; i < 48; i++)
    words[i] = words[i - 16] +
      (cgpt_sha256_rotate(words[i - 15], 7) ^ cgpt_sha256_rotate(words[i - 15], 18) ^
       (words[i - 15] >> 3)) +
               words[i - 7] +
      (cgpt_sha256_rotate(words[i - 2], 17) ^ cgpt_sha256_rotate(words[i - 2], 19) ^
       (words[i - 2] >> 10));
  // sixth round
  x = h + cgpt_sha256_f1(e, f, g) + 0xa2bfe8a1 + words[40];
  y = cgpt_sha256_f2(a, b, c);
  d += x;
  h = x + y;
  x = g + cgpt_sha256_f1(d, e, f) + 0xa81a664b + words[41];
  y = cgpt_sha256_f2(h, a, b);
  c += x;
  g = x + y;
  x = f + cgpt_sha256_f1(c, d, e) + 0xc24b8b70 + words[42];
  y = cgpt_sha256_f2(g, h, a);
  b += x;
  f = x + y;
  x = e + cgpt_sha256_f1(b, c, d) + 0xc76c51a3 + words[43];
  y = cgpt_sha256_f2(f, g, h);
  a += x;
  e = x + y;
  x = d + cgpt_sha256_f1(a, b, c) + 0xd192e819 + words[44];
  y = cgpt_sha256_f2(e, f, g);
  h += x;
  d = x + y;
  x = c + cgpt_sha256_f1(h, a, b) + 0xd6990624 + words[45];
  y = cgpt_sha256_f2(d, e, f);
  g += x;
  c = x + y;
  x = b + cgpt_sha256_f1(g, h, a) + 0xf40e3585 + words[46];
  y = cgpt_sha256_f2(c, d, e);
  f += x;
  b = x + y;
  x = a + cgpt_sha256_f1(f, g, h) + 0x106aa070 + words[47];
  y = cgpt_sha256_f2(b, c, d);
  e += x;
  a = x + y;
  // extend to 56 words
  for (; i < 56; i++)
    words[i] = words[i - 16] +
      (cgpt_sha256_rotate(words[i - 15], 7) ^ cgpt_sha256_rotate(words[i - 15], 18) ^
       (words[i - 15] >> 3)) +
               words[i - 7] +
      (cgpt_sha256_rotate(words[i - 2], 17) ^ cgpt_sha256_rotate(words[i - 2], 19) ^
       (words[i - 2] >> 10));
  // seventh round
  x = h + cgpt_sha256_f1(e, f, g) + 0x19a4c116 + words[48];
  y = cgpt_sha256_f2(a, b, c);
  d += x;
  h = x + y;
  x = g + cgpt_sha256_f1(d, e, f) + 0x1e376c08 + words[49];
  y = cgpt_sha256_f2(h, a, b);
  c += x;
  g = x + y;
  x = f + cgpt_sha256_f1(c, d, e) + 0x2748774c + words[50];
  y = cgpt_sha256_f2(g, h, a);
  b += x;
  f = x + y;
  x = e + cgpt_sha256_f1(b, c, d) + 0x34b0bcb5 + words[51];
  y = cgpt_sha256_f2(f, g, h);
  a += x;
  e = x + y;
  x = d + cgpt_sha256_f1(a, b, c) + 0x391c0cb3 + words[52];
  y = cgpt_sha256_f2(e, f, g);
  h += x;
  d = x + y;
  x = c + cgpt_sha256_f1(h, a, b) + 0x4ed8aa4a + words[53];
  y = cgpt_sha256_f2(d, e, f);
  g += x;
  c = x + y;
  x = b + cgpt_sha256_f1(g, h, a) + 0x5b9cca4f + words[54];
  y = cgpt_sha256_f2(c, d, e);
  f += x;
  b = x + y;
  x = a + cgpt_sha256_f1(f, g, h) + 0x682e6ff3 + words[55];
  y = cgpt_sha256_f2(b, c, d);
  e += x;
  a = x + y;
  // extend to 64 words
  for (; i < 64; i++)
    words[i] = words[i - 16] +
      (cgpt_sha256_rotate(words[i - 15], 7) ^ cgpt_sha256_rotate(words[i - 15], 18) ^
       (words[i - 15] >> 3)) +
               words[i - 7] +
      (cgpt_sha256_rotate(words[i - 2], 17) ^ cgpt_sha256_rotate(words[i - 2], 19) ^
       (words[i - 2] >> 10));
  // eigth round
  x = h + cgpt_sha256_f1(e, f, g) + 0x748f82ee + words[56];
  y = cgpt_sha256_f2(a, b, c);
  d += x;
  h = x + y;
  x = g + cgpt_sha256_f1(d, e, f) + 0x78a5636f + words[57];
  y = cgpt_sha256_f2(h, a, b);
  c += x;
  g = x + y;
  x = f + cgpt_sha256_f1(c, d, e) + 0x84c87814 + words[58];
  y = cgpt_sha256_f2(g, h, a);
  b += x;
  f = x + y;
  x = e + cgpt_sha256_f1(b, c, d) + 0x8cc70208 + words[59];
  y = cgpt_sha256_f2(f, g, h);
  a += x;
  e = x + y;
  x = d + cgpt_sha256_f1(a, b, c) + 0x90befffa + words[60];
  y = cgpt_sha256_f2(e, f, g);
  h += x;
  d = x + y;
  x = c + cgpt_sha256_f1(h, a, b) + 0xa4506ceb + words[61];
  y = cgpt_sha256_f2(d, e, f);
  g += x;
  c = x + y;
  x = b + cgpt_sha256_f1(g, h, a) + 0xbef9a3f7 + words[62];
  y = cgpt_sha256_f2(c, d, e);
  f += x;
  b = x + y;
  x = a + cgpt_sha256_f1(f, g, h) + 0xc67178f2 + words[63];
  y = cgpt_sha256_f2(b, c, d);
  e += x;
  a = x + y;
  // update hash
  newHash[0] = a + oldHash[0];
  newHash[1] = b + oldHash[1];
  newHash[2] = c + oldHash[2];
  newHash[3] = d + oldHash[3];
  newHash[4] = e + oldHash[4];
  newHash[5] = f + oldHash[5];
  newHash[6] = g + oldHash[6];
  newHash[7] = h + oldHash[7];
}

static void cgpt_sha256_processInput(uint32_t hash[8], const uint32_t oldHash[8],
				     const uint64_t numBytes, const uint8_t* input,
				     const size_t inputSize)
// process final block, less than 64 bytes
// newHash and oldHash and be the same
{
  // the input bytes are considered as bits strings, where the first bit is the
  // most significant bit of the byte
  // - append "1" bit to message
  // - append "0" bits until message length in bit mod 512 is 448
  // - append length as 64 bit integer
  // process initial parts of input
  std::memmove(hash, oldHash, 32);
  const int nBlocks = inputSize / 64;
  for (int i = 0; i < nBlocks; ++i) {
    cgpt_sha256_processBlock(hash, hash, input + i * 64);
  }
  // initialize buffer from input
  const size_t bufferSize = inputSize - nBlocks * 64;
  unsigned char buffer[cgpt_sha256_BlockSize];
  std::memcpy(buffer, input + nBlocks * 64, bufferSize);
  // number of bits
  size_t paddedLength = bufferSize * 8;
  // plus one bit set to 1 (always appended)
  paddedLength++;
  // number of bits must be (numBits % 512) = 448
  size_t lower11Bits = paddedLength & 511;
  if (lower11Bits <= 448) {
    paddedLength += 448 - lower11Bits;
  } else {
    paddedLength += 512 + 448 - lower11Bits;
  }
  // convert from bits to bytes
  paddedLength /= 8;
  // only needed if additional data flows over into a second block
  unsigned char extra[cgpt_sha256_BlockSize];
  // append a "1" bit, 128 => binary 10000000
  if (bufferSize < cgpt_sha256_BlockSize) {
    buffer[bufferSize] = 128;
  } else {
    extra[0] = 128;
  }
  size_t i;
  for (i = bufferSize + 1; i < cgpt_sha256_BlockSize; i++) {
    buffer[i] = 0;
  }
  for (; i < paddedLength; i++) {
    extra[i - cgpt_sha256_BlockSize] = 0;
  }
  // add message length in bits as 64 bit number
  uint64_t msgBits = 8 * (numBytes + inputSize);
  // find right position
  unsigned char* addLength;
  if (paddedLength < cgpt_sha256_BlockSize) {
    addLength = buffer + paddedLength;
  } else {
    addLength = extra + paddedLength - cgpt_sha256_BlockSize;
  }
  // must be big endian
  *addLength++ = (unsigned char)((msgBits >> 56) & 0xFF);
  *addLength++ = (unsigned char)((msgBits >> 48) & 0xFF);
  *addLength++ = (unsigned char)((msgBits >> 40) & 0xFF);
  *addLength++ = (unsigned char)((msgBits >> 32) & 0xFF);
  *addLength++ = (unsigned char)((msgBits >> 24) & 0xFF);
  *addLength++ = (unsigned char)((msgBits >> 16) & 0xFF);
  *addLength++ = (unsigned char)((msgBits >> 8) & 0xFF);
  *addLength = (unsigned char)(msgBits & 0xFF);
  // process blocks
  cgpt_sha256_processBlock(hash, hash, buffer);
  // flowed over into a second block ?
  if (paddedLength > cgpt_sha256_BlockSize) {
    cgpt_sha256_processBlock(hash, hash, extra);
  }
}

static void cgpt_sha256(uint32_t hash[8], const void* data, const size_t size) {
  uint32_t initHash[8];
  cgpt_sha256_setInitialHash(initHash);
  cgpt_sha256_processInput(hash, initHash, 0, (const uint8_t*)data, size);
}
