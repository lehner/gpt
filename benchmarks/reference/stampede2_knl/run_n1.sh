#!/bin/bash
OMP_NUM_THREADS=64 mpirun -n 1 $@ --mpi 1.1.1.1 --comms-overlap --dslash-asm --grid 24.24.24.24 --comms-concurrent
