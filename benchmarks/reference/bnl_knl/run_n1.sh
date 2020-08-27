#!/bin/bash
OMP_NUM_THREADS=64 ./dslash.py --grid 24.24.24.24 --dslash-asm --N 100
OMP_NUM_THREADS=64 ./dslash.py --grid 32.32.32.32 --dslash-asm --N 100

