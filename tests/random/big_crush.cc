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
#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "unif01.h"
#include "bbattery.h"
};

extern void* cgpt_rng_test_create(int iengine);
extern void cgpt_rng_test_destroy(void* t);
extern double cgpt_rng_test_GetU01(void* param, void* state);
extern unsigned long cgpt_rng_test_GetBits(void* param, void* state);
extern void cgpt_rng_test_Write(void* state);

int main(int argc, char* argv[]) {
  for (int iengine=0;;iengine++) {
    void* p = cgpt_rng_test_create(iengine);
    if (!p)
      break;
    unif01_Gen g = { p, p, "cgpt_ranlux", cgpt_rng_test_GetU01, cgpt_rng_test_GetBits, cgpt_rng_test_Write };
    printf("\n\n================================================================================\n\n");
    printf(" Run tests for cgpt engine = %d\n",iengine);
    printf("\n\n================================================================================\n\n");
    bbattery_SmallCrush (&g);
    bbattery_BigCrush (&g);
    cgpt_rng_test_destroy(p);
  }
  return 0;
}
