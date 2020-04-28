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
#define SHRT_UMAX 65535
#define FP16_BASE 1.4142135623730950488
#define FP16_COEF_EXP_SHARE_FLOATS 10

static float unmap_fp16_exp(uint16_t e) {
  float de = (float)((int)e - SHRT_UMAX / 2);
  return ::pow( FP16_BASE, de );
}

// can assume that v >=0 and need to guarantee that unmap_fp16_exp(map_fp16_exp(v)) >= v
static unsigned short map_fp16_exp(float v) {
  // float has exponents 10^{-44.85} .. 10^{38.53}
  int exp = (int)ceil(::log(v) / ::log(FP16_BASE)) + SHRT_UMAX / 2;
  if (exp < 0 || exp > SHRT_UMAX) {
    ERR("Error in map_fp16_exp(%g,%d)\n",v,exp);
  }
  
  return (unsigned short)exp;
}

accelerator_inline float fp_unmap(int val, float min, float max, int N) {
  return min + (float)(val + 0.5) * (max - min)  / (float)( N + 1 );
}

accelerator_inline int fp_map(float in, float min, float max, int N) {
  // Idea:
  //
  // min=-6
  // max=6
  //
  // N=1
  // [-6,0] -> 0, [0,6] -> 1;  reconstruct 0 -> -3, 1-> 3
  //
  // N=2
  // [-6,-2] -> 0, [-2,2] -> 1, [2,6] -> 2;  reconstruct 0 -> -4, 1->0, 2->4
  int ret =  (int) ( (float)(N+1) * ( (in - min) / (max - min) ) );
  if (ret == N+1) {
    ret = N;
  }
  return ret;
}

